"""The Value Analyst: Buffett's owner earnings and their value, made without seeing the price."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.analysts import value_analyst as va
from tradingagents.dataflows import valuation_data


def _row(ocf=100e6, capex=30e6, da=20e6, sbc=10e6, revenue=1000e6, **extra):
    return {"Operating Cash Flow": ocf, "Capital Expenditure": capex, "Depreciation & Amortization": da,
            "Stock Compensation": sbc, "Revenue": revenue, "Net Income": 60e6,
            "Stockholders Equity": 400e6, **extra}


def _rows(n=6, **kw):
    return {f"{2015 + i}-12-31": _row(revenue=1000e6 * 1.05 ** i, **kw) for i in range(n)}


def _growing(rate, n=6):
    """Rows whose owner earnings grow at ``rate`` while revenue stays flat."""
    return {f"{2015 + i}-12-31": _row(ocf=60e6 * (1 + rate) ** i + 30e6, revenue=1000e6) for i in range(n)}


@pytest.mark.unit
def test_owner_earnings_count_stock_pay_and_bound_maintenance_by_depreciation():
    row = _row()
    assert va.owner_earnings(row, base=False) == 100e6 - 10e6 - 30e6   # every dollar of capex
    assert va.owner_earnings(row, base=True) == 100e6 - 10e6 - 20e6    # no more than depreciation
    assert va.owner_earnings({**row, "Capital Expenditure": None}, base=True) is None


@pytest.mark.unit
def test_with_no_growth_the_value_is_earnings_over_the_rate():
    assert va.discounted_value(60e6, 0.0, 0.06, 0.0) == pytest.approx(1e9)


@pytest.mark.unit
def test_a_steady_business_is_valued_at_the_floored_rate_and_capped_growth():
    v = va.assess(_rows(), treasury=0.03)
    assert v.too_hard is None and v.discount == va.DISCOUNT_FLOOR
    assert v.growth == 0.0                               # revenue grows 5% a year; owner earnings do not
    assert v.value[0] == pytest.approx(60e6 / 0.06)      # conservative: (100 - 10 - 30)M held forever
    assert v.value[1] > v.value[0]
    assert va.assess(_rows(), treasury=0.07).discount == 0.07


@pytest.mark.unit
@pytest.mark.parametrize("rows,sic,reason", [
    (_rows(), "6021", "bank"),
    (_rows(n=4), "3571", "only 4 years"),
    (_rows(ocf=10e6), "3571", "negative"),
    ({**_rows(), "2020-12-31": _row(capex=None)}, "3571", "standard tag"),
])
def test_what_owner_earnings_cannot_value_is_too_hard(rows, sic, reason):
    v = va.assess(rows, treasury=0.04, sic=sic)
    assert reason in v.too_hard and v.value is None


class _Model:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage("1. Reinvestment needs: B")


def _state(asset_type="stock"):
    return {"company_of_interest": "ACME", "trade_date": "2024-06-03", "asset_type": asset_type}


@pytest.mark.unit
def test_the_model_sees_neither_name_nor_price_and_its_grades_are_marked_low_trust(monkeypatch):
    monkeypatch.setattr(valuation_data, "annual_filings", lambda *a: (_rows(), ("2024-04-30", 10e6), "3571"))
    monkeypatch.setattr(valuation_data, "treasury_10y", lambda *a: 0.044)
    monkeypatch.setattr(valuation_data, "price_and_split_factor", lambda *a: (987.65, 2.0))
    model = _Model()

    report = va.create_value_analyst(model)(_state())["valuation_report"]

    prompt = model.prompts[0]
    assert "ACME" not in prompt and "987.65" not in prompt and "a share" not in prompt
    assert va.LOW_TRUST in report and report.index("Qualitative grades") < report.index("Value vs. price")
    assert "Price 987.65, shares 20M" in report    # the split factor applied to the cover-page count


@pytest.mark.unit
def test_without_filings_or_for_crypto_no_model_call_is_made(monkeypatch):
    def none(*a):
        raise RuntimeError("not a US SEC filer")
    monkeypatch.setattr(valuation_data, "annual_filings", none)
    model = _Model()
    node = va.create_value_analyst(model)
    assert "not a US SEC filer" in node(_state())["valuation_report"]
    assert "Not available" in node(_state(asset_type="crypto"))["valuation_report"]
    assert model.prompts == []


@pytest.mark.unit
def test_a_price_above_value_reads_as_a_multiple_not_a_negative_margin():
    v = va.assess(_rows(), treasury=0.04)
    text = va.render_price(v, price=v.value[1] / 1e6 * 2, shares=1e6)
    assert "2.0x this value" in text and "margin of safety -" not in text


@pytest.mark.unit
def test_without_a_cover_count_the_latest_filed_share_total_is_used_on_its_filing_date_basis():
    gaap = {"CommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2023-12-31", "filed": "2024-02-01", "val": 100.0},
        {"end": "2024-03-31", "filed": "2024-04-26", "val": 98.0},
        {"end": "2024-06-30", "filed": "2024-07-25", "val": 97.0},   # filed after the analysis date
    ]}}}
    assert valuation_data._filed_shares(gaap, "2024-06-03") == ("2024-04-26", 98.0)
    assert valuation_data._filed_shares({}, "2024-06-03") is None
    # News Corp's last count of this kind was filed in 2015: too old to price a 2026 run.
    assert valuation_data._filed_shares(gaap, "2026-10-24") is None


@pytest.mark.unit
def test_with_no_recent_share_count_the_valuation_stands_and_only_the_price_comparison_is_dropped(monkeypatch):
    monkeypatch.setattr(valuation_data, "annual_filings", lambda *a: (_rows(), None, "2711"))
    monkeypatch.setattr(valuation_data, "treasury_10y", lambda *a: 0.044)
    report = va.create_value_analyst(_Model())(_state())["valuation_report"]
    assert "Base value" in report and "no share count filed in the 15 months" in report


@pytest.mark.unit
def test_growth_follows_owner_earnings_not_revenue_and_is_capped():
    assert va.owner_earnings_growth([100, 100, 100, 121, 121, 121]) == pytest.approx(0.0656, abs=1e-3)
    assert va.owner_earnings_growth([100, 100, 100, 400, 400, 400]) == va.GROWTH_CAP
    assert va.owner_earnings_growth([-5, -5, 20, 30]) == 0.0   # no positive start to grow from
    assert va.assess(_growing(0.04), treasury=0.04).growth == pytest.approx(0.04, abs=0.01)


@pytest.mark.unit
def test_untagged_gross_profit_and_operating_income_are_computed_and_marked():
    row = {"Revenue": 1000.0, "Gross Profit": None, "Operating Income": None, "Depreciation & Amortization": 30.0}
    valuation_data.fill_derived(row, cost=None, cost_ex_da=700.0, opex=200.0, da_apart=True)
    assert row["Gross Profit"] == 300.0 and row["Operating Income"] == 70.0
    assert row["_derived"] == {"Gross Profit", "Operating Income"}

    filed = {"Revenue": 1000.0, "Gross Profit": 320.0, "Operating Income": 90.0}
    valuation_data.fill_derived(filed, cost=700.0, cost_ex_da=None, opex=200.0, da_apart=False)
    assert filed["Gross Profit"] == 320.0 and filed["_derived"] == set()    # filed figures are kept


@pytest.mark.unit
def test_a_computed_cell_is_starred_and_blanks_are_explained():
    rows = _rows()
    rows["2020-12-31"]["Gross Profit"], rows["2020-12-31"]["_derived"] = 250e6, {"Gross Profit"}
    text = va.render_numbers(va.assess(rows, treasury=0.04))
    assert "250*" in text and "not that the company kept it undisclosed" in text
