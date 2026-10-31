"""Inputs for the price-blind Buffett valuation: annual filings, shares, the Treasury yield, the price.

Filings come from SEC EDGAR as they stood on the analysis date (see
``vendors.sec_edgar``). The price is fetched separately, and only after the
valuation is written, so nothing the valuer reads carries it.
"""

from __future__ import annotations

from datetime import date, timedelta

import yfinance as yf

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors import sec_edgar as edgar
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_INCOME = dict(edgar._STATEMENTS["income_statement"])
_CASHFLOW = dict(edgar._STATEMENTS["cashflow"])
_BALANCE = dict(edgar._STATEMENTS["balance_sheet"])

# Each line with the tags filers use for it, best first (first match wins, as in sec_edgar).
LINES: dict[str, tuple[str, ...]] = {
    "Revenue": _INCOME["Revenue"],
    "Gross Profit": _INCOME["Gross Profit"],
    "Operating Income": _INCOME["Operating Income"],
    "Net Income": _INCOME["Net Income"],
    "Operating Cash Flow": _CASHFLOW["Operating Cash Flow"],
    "Capital Expenditure": _CASHFLOW["Capital Expenditure"],
    "Depreciation & Amortization": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                                    "DepreciationAmortizationAndAccretionNet", "Depreciation"),
    "Stock Compensation": ("ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"),
    "Buybacks": ("PaymentsForRepurchaseOfCommonStock",),
    "Dividends": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    "Total Assets": _BALANCE["Total Assets"],
    "Total Liabilities": _BALANCE["Total Liabilities"],
    "Stockholders Equity": _BALANCE["Stockholders Equity"],
}

# Lines read only to compute Gross Profit and Operating Income for a filer that
# does not tag them (Casey's has tagged neither since 2020).
_COST = (*_INCOME["Cost of Revenue"], "CostOfGoodsSold")
_COST_EX_DA = ("CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",)
_OPEX = ("OperatingExpenses",)


def fill_derived(row: dict, cost: float | None, cost_ex_da: float | None, opex: float | None,
                 da_apart: bool) -> None:
    """Compute Gross Profit and Operating Income from the lines a filer did tag, naming them in ``_derived``.

    ``da_apart``: the filer states cost of goods without depreciation (in any
    year), so depreciation is a line of its own and comes off operating income.
    """
    derived = row.setdefault("_derived", set())
    if row.get("Gross Profit") is None and row.get("Revenue") is not None:
        if cost is not None or cost_ex_da is not None:
            row["Gross Profit"] = row["Revenue"] - (cost if cost is not None else cost_ex_da)
            derived.add("Gross Profit")
    if row.get("Operating Income") is None and row.get("Gross Profit") is not None and opex is not None:
        depreciation = (row.get("Depreciation & Amortization") or 0) if da_apart else 0
        row["Operating Income"] = row["Gross Profit"] - opex - depreciation
        derived.add("Operating Income")


# A share count older than this (a year's 10-K plus its filing lag) is not today's.
_FRESH_DAYS = 456


def _fresh(day: str, as_of_date: str) -> bool:
    return (date.fromisoformat(as_of_date) - date.fromisoformat(day)).days <= _FRESH_DAYS


def annual_filings(ticker: str, as_of_date: str,
                   years: int = 10) -> tuple[dict[str, dict], tuple[str, float] | None, str]:
    """({fiscal year end: {line: value or None}}, (basis date, shares outstanding) or None, SIC industry code).

    Up to ``years`` fiscal years from 10-K filings as filed by ``as_of_date``,
    oldest first. Shares come from the latest filing's cover page, so their split
    basis is that date's; None when no count was filed in the last 15 months.
    """
    cik = edgar.cik_for(ticker)
    if cik is None:
        raise NoMarketDataError(ticker, ticker, "not a US SEC filer")
    facts = edgar._cached_json(edgar._FACTS_URL.format(cik=cik), f"CIK{cik}.json").get("facts") or {}
    gaap = facts.get("us-gaap") or {}

    values = {label: edgar._as_of(gaap, tags, as_of_date, (edgar._SPANS["annual"],), edgar._ANNUAL_FORMS)[0]
              for label, tags in LINES.items()}
    ends = sorted(end for end, _ in values["Net Income"] if (end, 0) in values["Operating Cash Flow"])[-years:]
    if not ends:
        raise NoMarketDataError(ticker, ticker, f"no annual filings by {as_of_date}")
    rows = {end: {label: v.get((end, 0)) for label, v in values.items()} for end in ends}
    cost, cost_ex_da, opex = (edgar._as_of(gaap, tags, as_of_date, (edgar._SPANS["annual"],), edgar._ANNUAL_FORMS)[0]
                              for tags in (_COST, _COST_EX_DA, _OPEX))
    for end, row in rows.items():
        fill_derived(row, cost.get((end, 0)), cost_ex_da.get((end, 0)), opex.get((end, 0)), bool(cost_ex_da))

    # The cover page reports shares as instants, which the span check passes as index 0.
    covers = edgar._as_of(facts.get("dei") or {}, ("EntityCommonStockSharesOutstanding",), as_of_date, ((0, 0),))[0]
    cover_end = max((end for end, _ in covers), default=None)
    if cover_end and _fresh(cover_end, as_of_date):
        shares = (cover_end, float(covers[(cover_end, 0)]))
    else:
        shares = _filed_shares(gaap, as_of_date)
    sic = str(edgar._cached_json(_SUBMISSIONS_URL.format(cik=cik), f"submissions_CIK{cik}.json").get("sic") or "")
    return rows, shares, sic


def _filed_shares(gaap: dict, as_of_date: str) -> tuple[str, float] | None:
    """(filing date, shares) from the latest statement filed, for filers with no cover-page count.

    SEC's company facts leave out a cover page that counts each share class
    separately (Alphabet's, for one); the balance sheet's total, else the diluted
    average, stands in. A filing states share counts on the split basis of the
    day it was filed, so the filing date is the basis date.
    """
    for tag in ("CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"):
        facts = [f for f in (gaap.get(tag) or {}).get("units", {}).get("shares", []) if f["filed"] <= as_of_date]
        if facts:
            latest = max(facts, key=lambda f: (f["filed"], f["end"]))
            if _fresh(latest["filed"], as_of_date):
                return latest["filed"], float(latest["val"])
    return None


def treasury_10y(as_of_date: str) -> float:
    """The 10-year Treasury yield (as a fraction) at the last close on or before ``as_of_date``."""
    return float(load_ohlcv("^TNX", as_of_date)["Close"].iloc[-1]) / 100


def price_and_split_factor(ticker: str, as_of_date: str, since: str) -> tuple[float, float]:
    """(close on or before ``as_of_date``, splits since ``since`` multiplied together).

    Yahoo states every past close on today's split basis, so shares counted on
    ``since`` are multiplied by the factor to match it. The close is not adjusted
    for dividends: a dividend-adjusted past close understates what it traded at.
    """
    t = yf.Ticker(ticker)
    day = date.fromisoformat(as_of_date)
    history = t.history(start=day - timedelta(days=10), end=day + timedelta(days=1), auto_adjust=False)
    if history.empty:
        raise NoMarketDataError(ticker, ticker, f"no close on or before {as_of_date}")
    factor = 1.0
    for when, ratio in t.splits.items():
        if when.date() > date.fromisoformat(since):
            factor *= float(ratio)
    return float(history["Close"].iloc[-1]), factor
