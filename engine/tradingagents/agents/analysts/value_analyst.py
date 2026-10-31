"""Value Analyst: a Buffett-style intrinsic value, made from the filings without seeing the price.

Code computes everything a formula can, following Buffett's own definitions:
owner earnings (1986 letter: cash an owner could take out, after the capital
spending needed to stay competitive) and their value discounted at the 10-year
Treasury rate (Owner's Manual). The model, shown only those numbers, with no
name, ticker or price, grades what needs judgment A-F. Those grades are marked
low-trust in the report for every agent that reads it. The price is compared
with the value only after the valuation is written.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tradingagents.agents.context import get_language_instruction
from tradingagents.dataflows import valuation_data

logger = logging.getLogger(__name__)

# Buffett discounts at the long-term Treasury rate and asks a margin of safety
# instead of a risk premium. Below this floor a low-rate year values any steady
# business near infinity, so the rate used is never lower.
DISCOUNT_FLOOR = 0.06
GROWTH_CAP = 0.08          # base-case growth: owner-earnings growth on file, held to 0..8% a year
TERMINAL_GROWTH = 0.02     # base case after ten years; the conservative case assumes none
YEARS = 10

LOW_TRUST = (
    "LOW TRUST: a model's best guess from the numbers above alone, with no company name, filings text or "
    "management record. Weigh these grades far below the computed figures: a tie-breaker, never the reason for a call."
)


@dataclass
class Valuation:
    rows: dict[str, dict]           # fiscal year end -> line -> value, with owner earnings added
    treasury: float | None
    discount: float
    growth: float
    owner_earnings: tuple[float, float] | None = None   # normalized (conservative, base)
    value: tuple[float, float] | None = None            # equity value (conservative, base), USD
    too_hard: str | None = None


def owner_earnings(row: dict, base: bool) -> float | None:
    """Operating cash flow - stock compensation - maintenance capex.

    Maintenance capex is not reported. The conservative case counts all capex as
    maintenance; the base case counts no more than depreciation, the spending that
    replaces what wore out.
    """
    ocf, capex = row.get("Operating Cash Flow"), row.get("Capital Expenditure")
    if ocf is None or capex is None:
        return None
    depreciation = row.get("Depreciation & Amortization")
    maintenance = min(capex, depreciation) if base and depreciation is not None else capex
    return ocf - (row.get("Stock Compensation") or 0) - maintenance


def discounted_value(earnings: float, growth: float, discount: float, terminal: float) -> float:
    """Ten years of earnings growing at ``growth``, then forever at ``terminal``, discounted."""
    years = sum(earnings * (1 + growth) ** t / (1 + discount) ** t for t in range(1, YEARS + 1))
    tail = earnings * (1 + growth) ** YEARS * (1 + terminal) / (discount - terminal) / (1 + discount) ** YEARS
    return years + tail


def owner_earnings_growth(earnings: list[float]) -> float:
    """Yearly growth from the first three years' average to the last three's, held to 0..GROWTH_CAP.

    Owner earnings, not revenue: a fuel seller's revenue moves with the price of
    fuel, not with what the business earns. Totals, not per share: filed share
    counts jump at every split, which would read as growth or dilution.
    """
    k = min(3, len(earnings) // 2)
    if not k:
        return 0.0
    start, end = sum(earnings[:k]) / k, sum(earnings[-k:]) / k
    if start <= 0 or end <= 0:
        return 0.0
    return min(max((end / start) ** (1 / (len(earnings) - k)) - 1, 0.0), GROWTH_CAP)


def assess(rows: dict[str, dict], treasury: float | None, sic: str = "") -> Valuation:
    """Owner earnings for every year and their value, or why the method does not fit."""
    for row in rows.values():
        row["Owner Earnings (conservative)"] = owner_earnings(row, base=False)
        row["Owner Earnings (base)"] = owner_earnings(row, base=True)
    discount = max(treasury or 0.0, DISCOUNT_FLOOR)
    base = [r["Owner Earnings (base)"] for r in rows.values() if r["Owner Earnings (base)"] is not None]
    v = Valuation(rows=rows, treasury=treasury, discount=discount, growth=owner_earnings_growth(base))

    # SIC 6000-6499: banks, lenders, brokers and insurers, whose operating cash flow
    # moves with deposits and claims rather than with what the business earns.
    if sic.isdigit() and 6000 <= int(sic) <= 6499:
        v.too_hard = f"a bank, lender, broker or insurer (SIC {sic}); owner earnings from cash flow do not value one"
    elif len(rows) < 5:
        v.too_hard = f"only {len(rows)} years of annual filings; at least 5 are needed to judge a record"
    # Some filers tag capex under their own name for years; those years stay blank,
    # but the three-year average needs the last three.
    elif any(r["Owner Earnings (base)"] is None for r in list(rows.values())[-3:]):
        v.too_hard = "capital spending is not filed under a standard tag for the last three years"
    elif sum(e <= 0 for e in base) >= 3:
        v.too_hard = f"owner earnings were negative in {sum(e <= 0 for e in base)} of {len(base)} years"
    if v.too_hard:
        return v

    recent = list(rows.values())[-3:]
    normal = tuple(sum(r[k] or 0 for r in recent) / len(recent)
                   for k in ("Owner Earnings (conservative)", "Owner Earnings (base)"))
    if normal[1] <= 0:
        v.too_hard = "owner earnings averaged zero or less over the last three years"
        return v
    v.owner_earnings = normal
    v.value = (max(normal[0], 0.0) / discount,      # no growth, ever: what the business earns today, held
               discounted_value(normal[1], v.growth, discount, TERMINAL_GROWTH))
    return v


def _m(x: float | None) -> str:
    return "" if x is None else f"{x / 1e6:,.0f}"


def render_numbers(v: Valuation) -> str:
    """The filing table and the computed value, with no name, ticker, per-share figure or price."""
    years = list(v.rows)
    lines = ["| USD millions | " + " | ".join(y[:4] for y in years) + " |",
             "|---|" + "---|" * len(years)]
    labels = [*valuation_data.LINES, "Owner Earnings (conservative)", "Owner Earnings (base)"]
    for label in labels:
        cells = [_m(v.rows[y].get(label)) + ("*" if label in v.rows[y].get("_derived", ()) else "")
                 for y in years]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for name, num, den in (("Gross margin", "Gross Profit", "Revenue"),
                           ("Operating margin", "Operating Income", "Revenue"),
                           ("Return on equity", "Net Income", "Stockholders Equity")):
        cells = []
        for y in years:
            a, b = v.rows[y].get(num), v.rows[y].get(den)
            cells.append(f"{a / b:.0%}" if a is not None and b else "")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    treasury = f"{v.treasury:.2%}" if v.treasury is not None else "unavailable"
    out = ["### Filing data (SEC EDGAR, annual reports as filed by the analysis date)", *lines, "",
           "\\* computed from other filed lines; the company does not tag this one. A blank cell means the line is "
           "not in the standard tags SEC data uses, not that the company kept it undisclosed.", "",
           "### Intrinsic value (computed)",
           f"- Discount rate: {v.discount:.2%} (10-year Treasury {treasury}, floored at {DISCOUNT_FLOOR:.0%})"]
    if v.too_hard:
        out.append(f"- **Too hard to value:** {v.too_hard}. No intrinsic value is given.")
        return "\n".join(out)
    cons, base = v.owner_earnings
    out += [
        f"- Normalized owner earnings (3-year average): conservative {_m(cons)}M, base {_m(base)}M",
        f"- Conservative value: {_m(v.value[0])}M (no growth: today's owner earnings held forever)",
        f"- Base value: {_m(v.value[1])}M ({v.growth:.1%} a year for {YEARS} years, the owner-earnings growth on "
        f"file held to 0-{GROWTH_CAP:.0%}, then {TERMINAL_GROWTH:.0%} a year)",
    ]
    return "\n".join(out)


def render_price(v: Valuation, price: float, shares: float) -> str:
    """Value against price, written after the valuation."""
    cap = price * shares
    lines = ["### Value vs. price (computed after the valuation was made)",
             f"- Price {price:,.2f}, shares {shares / 1e6:,.0f}M, market value {_m(cap)}M"]
    for name, value in zip(("Conservative", "Base"), v.value):
        if value <= 0:
            lines.append(f"- {name}: no positive value")
        elif cap <= value:
            lines.append(f"- {name}: {value / shares:,.2f} a share, margin of safety {(value - cap) / value:.0%}")
        else:
            lines.append(f"- {name}: {value / shares:,.2f} a share; the price is {cap / value:.1f}x this value "
                         "(no margin of safety)")
    lines.append("- Buffett buys only with a wide margin of safety; a positive margin under about 25% is thin.")
    return "\n".join(lines)


def create_value_analyst(llm):
    def value_analyst_node(state):
        ticker, trade_date = state["company_of_interest"], state["trade_date"]
        if state.get("asset_type", "stock") != "stock":
            return {"valuation_report": "Not available: owner earnings apply to companies, not this asset."}
        try:
            rows, shares, sic = valuation_data.annual_filings(ticker, trade_date)
        except Exception as exc:  # no filings is "no valuation", not a failed run
            logger.warning("Value Analyst: no filings for %s: %s", ticker, exc)
            return {"valuation_report": f"Not available: {exc}"}
        try:
            treasury = valuation_data.treasury_10y(trade_date)
        except Exception as exc:
            logger.warning("Value Analyst: no Treasury yield for %s: %s", trade_date, exc)
            treasury = None
        v = assess(rows, treasury, sic)
        numbers = render_numbers(v)
        parts = ["## Buffett-Style Valuation (price-blind)", numbers]

        if not v.too_hard:
            prompt = f"""You are grading an anonymous company the way Warren Buffett would, from its annual filings alone. You do not know its name, its industry or its share price, and must not guess them.

{numbers}

Grade each of these A to F from the numbers above, one line each with the grade and the evidence behind it. A blank cell is a gap in the standard data tags, not a disclosure choice by the company: never let a blank lower a grade.
1. Reinvestment needs: how much of its earnings must go back into the business just to stay competitive. Compare capital spending with depreciation and with owner earnings.
2. Durable advantage (moat): do high, steady margins and returns on equity over the years suggest pricing power that lasts, or do they swing and fade?
3. Capital allocation by management: what it did with the cash: buybacks, dividends, debt, and whether returns held up as equity grew.

Then one line: whether this passes Buffett's filters of favorable long-term economics and sensible capital use, judged from these numbers only. Do not state a value; it is computed above.""" + get_language_instruction()
            grades = llm.invoke(prompt).content
            parts.append(f"### Qualitative grades\n> {LOW_TRUST}\n\n{grades}")

            try:
                if shares is None:
                    raise ValueError("no share count filed in the 15 months before the analysis date")
                basis_date, count = shares
                price, factor = valuation_data.price_and_split_factor(ticker, trade_date, basis_date)
                parts.append(render_price(v, price, count * factor))
            except Exception as exc:
                logger.warning("Value Analyst: no price for %s: %s", ticker, exc)
                parts.append(f"### Value vs. price\nUnavailable: {exc}")

        return {"valuation_report": "\n\n".join(parts)}

    return value_analyst_node
