"""
AI-Enhanced Valuation Assistant
-------------------------------
A Streamlit app for a straightforward FCFF discounted cash flow valuation.

Default company: Celsius Holdings, Inc. (ticker: CELH)

How to run:
    streamlit run ai_enhanced_valuation_assistant_celsius.py

Recommended packages:
    streamlit
    pandas
    numpy
    yfinance

The app works with manual inputs even if online financial data is unavailable.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:  # keeps the app usable if yfinance is not installed
    yf = None


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI-Enhanced Valuation Assistant",
    page_icon="📈",
    layout="wide",
)


# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------
def money(value: float, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.{decimals}f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:,.{decimals}f}M"
    return f"${value:,.{decimals}f}"


def percent(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def safe_number(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


# ------------------------------------------------------------
# Data retrieval helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_yfinance_data(ticker: str) -> Dict[str, object]:
    """Return recent financial statement data when available."""
    result: Dict[str, object] = {
        "company_name": ticker,
        "current_price": np.nan,
        "shares_outstanding": np.nan,
        "market_cap": np.nan,
        "total_debt": np.nan,
        "cash": np.nan,
        "beta": np.nan,
        "income_statement": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cash_flow": pd.DataFrame(),
        "error": None,
    }

    if yf is None:
        result["error"] = "The yfinance package is not installed, so the app is using manual inputs."
        return result

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        result["company_name"] = info.get("longName") or info.get("shortName") or ticker
        result["current_price"] = safe_number(
            info.get("currentPrice") or info.get("regularMarketPrice"), np.nan
        )
        result["shares_outstanding"] = safe_number(info.get("sharesOutstanding"), np.nan)
        result["market_cap"] = safe_number(info.get("marketCap"), np.nan)
        result["total_debt"] = safe_number(info.get("totalDebt"), np.nan)
        result["cash"] = safe_number(info.get("totalCash"), np.nan)
        result["beta"] = safe_number(info.get("beta"), np.nan)

        result["income_statement"] = stock.financials.copy() if stock.financials is not None else pd.DataFrame()
        result["balance_sheet"] = stock.balance_sheet.copy() if stock.balance_sheet is not None else pd.DataFrame()
        result["cash_flow"] = stock.cashflow.copy() if stock.cashflow is not None else pd.DataFrame()

    except Exception as exc:
        result["error"] = f"Financial data could not be loaded: {exc}"

    return result


def first_available(statement: pd.DataFrame, possible_rows: list[str]) -> Optional[pd.Series]:
    """Find the first matching row in a yfinance statement."""
    if statement.empty:
        return None
    index_lookup = {str(idx).lower(): idx for idx in statement.index}
    for row in possible_rows:
        match = index_lookup.get(row.lower())
        if match is not None:
            return statement.loc[match].dropna()
    return None


def latest_value(statement: pd.DataFrame, possible_rows: list[str], default: float = np.nan) -> float:
    row = first_available(statement, possible_rows)
    if row is None or row.empty:
        return default
    return safe_number(row.iloc[0], default)


def historical_table(data: Dict[str, object]) -> pd.DataFrame:
    """Build a clean annual historical table from available statements."""
    income = data.get("income_statement", pd.DataFrame())
    cash_flow = data.get("cash_flow", pd.DataFrame())

    if income.empty:
        return pd.DataFrame()

    revenue = first_available(income, ["Total Revenue", "Revenue"])
    ebit = first_available(income, ["EBIT", "Operating Income"])
    net_income = first_available(income, ["Net Income", "Net Income Common Stockholders"])
    depreciation = first_available(cash_flow, ["Depreciation And Amortization", "Depreciation"])
    capex = first_available(cash_flow, ["Capital Expenditure", "Capital Expenditures"])

    rows = []
    columns = list(income.columns)
    for col in columns[:4]:
        rev = safe_number(revenue.get(col), np.nan) if revenue is not None else np.nan
        ebit_value = safe_number(ebit.get(col), np.nan) if ebit is not None else np.nan
        ni = safe_number(net_income.get(col), np.nan) if net_income is not None else np.nan
        da = safe_number(depreciation.get(col), 0.0) if depreciation is not None else 0.0
        capex_value = safe_number(capex.get(col), 0.0) if capex is not None else 0.0
        rows.append(
            {
                "Fiscal Year": str(pd.to_datetime(col).year),
                "Revenue": rev,
                "EBIT": ebit_value,
                "Net Income": ni,
                "D&A": abs(da),
                "CapEx": abs(capex_value),
                "EBIT Margin": safe_divide(ebit_value, rev),
            }
        )

    hist = pd.DataFrame(rows)
    return hist.dropna(how="all")


def compute_revenue_cagr(hist: pd.DataFrame) -> float:
    """Estimate recent revenue CAGR from oldest to newest available year."""
    if hist.empty or "Revenue" not in hist.columns:
        return np.nan
    cleaned = hist.dropna(subset=["Revenue"]).copy()
    cleaned = cleaned[cleaned["Revenue"] > 0]
    if len(cleaned) < 2:
        return np.nan

    # yfinance columns usually come newest first, so reverse to oldest first.
    newest = cleaned.iloc[0]["Revenue"]
    oldest = cleaned.iloc[-1]["Revenue"]
    periods = len(cleaned) - 1
    if oldest <= 0 or periods <= 0:
        return np.nan
    return (newest / oldest) ** (1 / periods) - 1


# ------------------------------------------------------------
# App header
# ------------------------------------------------------------
st.title("AI-Enhanced Valuation Assistant")
st.caption("A straightforward FCFF DCF model with historical inputs, manual assumptions, and an Excel-friendly walkthrough.")

st.markdown(
    """
    This app estimates an intrinsic value per share using a simple **free cash flow to the firm (FCFF)** approach.  
    It starts with recent company historicals when available, lets the user adjust the major assumptions, then converts enterprise value into equity value per share.
    """
)


# ------------------------------------------------------------
# Sidebar: company and assumptions
# ------------------------------------------------------------
st.sidebar.header("Company")
ticker = st.sidebar.text_input("Ticker", value="CELH").upper().strip() or "CELH"

# Historical financials are pulled automatically for the ticker entered above.
data = get_yfinance_data(ticker)
hist = historical_table(data)

company_name = data.get("company_name", ticker)
current_price = safe_number(data.get("current_price", np.nan), np.nan)
shares_auto = safe_number(data.get("shares_outstanding", np.nan), np.nan)
debt_auto = safe_number(data.get("total_debt", np.nan), np.nan)
cash_auto = safe_number(data.get("cash", np.nan), np.nan)
beta_auto = safe_number(data.get("beta", np.nan), np.nan)

latest_revenue = latest_value(data.get("income_statement", pd.DataFrame()), ["Total Revenue", "Revenue"], np.nan)
latest_ebit = latest_value(data.get("income_statement", pd.DataFrame()), ["EBIT", "Operating Income"], np.nan)
historical_ebit_margin = safe_divide(latest_ebit, latest_revenue)
historical_growth = compute_revenue_cagr(hist)

# Fallback defaults are only used if reported data is unavailable.
default_revenue = latest_revenue if not pd.isna(latest_revenue) and latest_revenue > 0 else 1_320_000_000.0
default_growth = historical_growth if not pd.isna(historical_growth) and -0.5 < historical_growth < 1.5 else 0.18
default_margin = historical_ebit_margin if not pd.isna(historical_ebit_margin) and -0.5 < historical_ebit_margin < 0.8 else 0.18
default_shares = shares_auto if not pd.isna(shares_auto) and shares_auto > 0 else 235_000_000.0
default_debt = debt_auto if not pd.isna(debt_auto) and debt_auto >= 0 else 0.0
default_cash = cash_auto if not pd.isna(cash_auto) and cash_auto >= 0 else 900_000_000.0
default_beta = beta_auto if not pd.isna(beta_auto) and beta_auto > 0 else 1.7

st.sidebar.header("Operating Forecast")
current_revenue = float(default_revenue)
st.sidebar.metric("Most recent annual revenue", money(current_revenue, 4))
revenue_growth = st.sidebar.number_input(
    "Annual revenue growth (%)",
    value=float(default_growth * 100),
    step=0.5,
    help="This is prefilled from the ticker's recent revenue CAGR when available, but you can adjust it for your forecast.",
) / 100
ebit_margin = st.sidebar.number_input(
    "EBIT margin (%)",
    min_value=-100.0,
    max_value=100.0,
    value=float(default_margin * 100),
    step=0.5,
    help="This is prefilled from the ticker's latest reported EBIT margin when available, but you can adjust it for your forecast.",
) / 100
tax_rate = st.sidebar.number_input(
    "Tax rate (%)",
    min_value=0.0,
    max_value=100.0,
    value=21.0,
    step=0.5,
    help="Tax rate applied to EBIT to calculate NOPAT.",
) / 100
reinvestment_rate = st.sidebar.number_input(
    "Reinvestment rate (% of NOPAT)",
    min_value=0.0,
    max_value=100.0,
    value=35.0,
    step=1.0,
    help="Simple estimate of the cash reinvested back into the business.",
) / 100
projection_years = st.sidebar.slider("Forecast years", min_value=3, max_value=10, value=5)

st.sidebar.header("WACC Inputs")
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%)",
    value=4.3,
    step=0.1,
    help="Often approximated with a long-term Treasury yield.",
) / 100
beta = st.sidebar.number_input(
    "Beta",
    min_value=0.0,
    value=float(default_beta),
    step=0.05,
    help="Measures how sensitive the stock is to the overall market.",
)
equity_risk_premium = st.sidebar.number_input(
    "Equity risk premium (%)",
    value=5.0,
    step=0.25,
    help="Extra return investors require for holding stocks instead of a risk-free asset.",
) / 100
cost_of_debt = st.sidebar.number_input(
    "Pre-tax cost of debt (%)",
    min_value=0.0,
    value=6.0,
    step=0.25,
    help="Estimated borrowing cost before tax adjustment.",
) / 100
equity_weight = st.sidebar.number_input(
    "Equity weight in capital structure (%)",
    min_value=0.0,
    max_value=100.0,
    value=95.0,
    step=1.0,
    help="Percent of financing from equity. Debt weight is calculated as 100% minus this number.",
) / 100
debt_weight = 1 - equity_weight

cost_of_equity = risk_free_rate + beta * equity_risk_premium
wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))

st.sidebar.header("Terminal Value and Equity Value")
terminal_growth = st.sidebar.number_input(
    "Perpetual growth rate (%)",
    min_value=-20.0,
    value=3.0,
    step=0.25,
    help="Long-run growth after the forecast period. This must be below WACC.",
) / 100
debt = float(default_debt)
cash = float(default_cash)
shares_outstanding = float(default_shares)
st.sidebar.metric("Total debt", money(debt, 4))
st.sidebar.metric("Cash and short-term investments", money(cash, 4))
st.sidebar.metric("Shares outstanding", f"{shares_outstanding:,.0f}")


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
if terminal_growth >= wacc:
    st.error("The perpetual growth rate must be lower than WACC for the terminal value formula to work.")
    st.stop()


# ------------------------------------------------------------
# DCF calculations
# ------------------------------------------------------------
years = np.arange(1, projection_years + 1)
rows = []
revenue = current_revenue

for year in years:
    revenue = revenue * (1 + revenue_growth)
    ebit = revenue * ebit_margin
    nopat = ebit * (1 - tax_rate)
    reinvestment = nopat * reinvestment_rate
    fcff = nopat - reinvestment
    discount_factor = 1 / ((1 + wacc) ** year)
    pv_fcff = fcff * discount_factor

    rows.append(
        {
            "Year": year,
            "Revenue": revenue,
            "EBIT": ebit,
            "NOPAT": nopat,
            "Reinvestment": reinvestment,
            "FCFF": fcff,
            "Discount Factor": discount_factor,
            "PV of FCFF": pv_fcff,
        }
    )

projection = pd.DataFrame(rows)
final_fcff = projection.iloc[-1]["FCFF"]
next_year_fcff = final_fcff * (1 + terminal_growth)
terminal_value = next_year_fcff / (wacc - terminal_growth)
pv_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
pv_forecast_fcff = projection["PV of FCFF"].sum()
enterprise_value = pv_forecast_fcff + pv_terminal_value
equity_value = enterprise_value - debt + cash
intrinsic_value_per_share = safe_divide(equity_value, shares_outstanding)

if not pd.isna(current_price) and current_price > 0:
    difference = intrinsic_value_per_share - current_price
    difference_pct = safe_divide(difference, current_price)
else:
    difference = np.nan
    difference_pct = np.nan


# ------------------------------------------------------------
# Main output
# ------------------------------------------------------------
st.header(f"Valuation for {company_name} ({ticker})")

if data.get("error"):
    st.info(data["error"])

summary_cols = st.columns(4)
summary_cols[0].metric("Intrinsic Value per Share", money(intrinsic_value_per_share))
summary_cols[1].metric("Enterprise Value", money(enterprise_value))
summary_cols[2].metric("Equity Value", money(equity_value))
summary_cols[3].metric("WACC", percent(wacc))

if not pd.isna(current_price) and current_price > 0:
    comp1, comp2, comp3 = st.columns(3)
    comp1.metric("Recent Market Price", money(current_price))
    comp2.metric("Model Difference", money(difference))
    comp3.metric("Difference %", percent(difference_pct))

    if intrinsic_value_per_share > current_price:
        st.success("The model estimate is above the recent market price based on the current assumptions.")
    elif intrinsic_value_per_share < current_price:
        st.warning("The model estimate is below the recent market price based on the current assumptions.")
    else:
        st.info("The model estimate is equal to the recent market price based on the current assumptions.")
else:
    st.info("A recent market price was not available. The app still calculates intrinsic value per share.")


# ------------------------------------------------------------
# Historical baseline
# ------------------------------------------------------------
st.header("Historical Baseline")
st.write(
    "The table below is used as a starting point for the forecast when recent financial statement data is available. "
    "The valuation still depends on the assumptions selected in the sidebar."
)

if not hist.empty:
    display_hist = hist.copy()
    for col in ["Revenue", "EBIT", "Net Income", "D&A", "CapEx"]:
        if col in display_hist.columns:
            display_hist[col] = display_hist[col].map(lambda x: money(x, 2))
    if "EBIT Margin" in display_hist.columns:
        display_hist["EBIT Margin"] = display_hist["EBIT Margin"].map(percent)
    st.dataframe(display_hist, use_container_width=True)

    baseline_cols = st.columns(3)
    baseline_cols[0].metric("Historical Revenue Growth", percent(historical_growth) if not pd.isna(historical_growth) else "N/A")
    baseline_cols[1].metric("Latest EBIT Margin", percent(historical_ebit_margin) if not pd.isna(historical_ebit_margin) else "N/A")
    baseline_cols[2].metric("Recent Market Price", money(current_price) if not pd.isna(current_price) else "N/A")
else:
    st.info("Historical statement data was not available, so the model is using the manual inputs in the sidebar.")


# ------------------------------------------------------------
# WACC breakdown
# ------------------------------------------------------------
st.header("WACC Breakdown")
wacc_table = pd.DataFrame(
    {
        "Input": [
            "Risk-free rate",
            "Beta",
            "Equity risk premium",
            "Cost of equity = Risk-free rate + Beta × Equity risk premium",
            "Pre-tax cost of debt",
            "Tax rate",
            "After-tax cost of debt = Cost of debt × (1 − Tax rate)",
            "Equity weight",
            "Debt weight",
            "WACC",
        ],
        "Value": [
            percent(risk_free_rate),
            f"{beta:.2f}",
            percent(equity_risk_premium),
            percent(cost_of_equity),
            percent(cost_of_debt),
            percent(tax_rate),
            percent(cost_of_debt * (1 - tax_rate)),
            percent(equity_weight),
            percent(debt_weight),
            percent(wacc),
        ],
    }
)
st.dataframe(wacc_table, use_container_width=True, hide_index=True)

st.latex(r"r_E = r_f + \beta \times ERP")
st.latex(r"WACC = w_E r_E + w_D r_D(1 - Tax\ Rate)")


# ------------------------------------------------------------
# Projection table
# ------------------------------------------------------------
st.header("DCF Projection")
projection_display = projection.copy()
for col in ["Revenue", "EBIT", "NOPAT", "Reinvestment", "FCFF", "PV of FCFF"]:
    projection_display[col] = projection_display[col].map(lambda x: money(x, 2))
projection_display["Discount Factor"] = projection_display["Discount Factor"].map(lambda x: f"{x:.4f}")
st.dataframe(projection_display, use_container_width=True, hide_index=True)

st.subheader("Forecast FCFF")
st.line_chart(projection.set_index("Year")["FCFF"])


# ------------------------------------------------------------
# Terminal value and equity bridge
# ------------------------------------------------------------
st.header("Terminal Value and Equity Bridge")
terminal_table = pd.DataFrame(
    {
        "Line Item": [
            "Final forecast year FCFF",
            "Next year FCFF = Final FCFF × (1 + perpetual growth)",
            "Terminal value = Next year FCFF ÷ (WACC − perpetual growth)",
            "PV of terminal value",
            "PV of forecast FCFF",
            "Enterprise value",
            "Less: debt",
            "Add: cash and short-term investments",
            "Equity value",
            "Shares outstanding",
            "Intrinsic value per share",
        ],
        "Amount": [
            money(final_fcff),
            money(next_year_fcff),
            money(terminal_value),
            money(pv_terminal_value),
            money(pv_forecast_fcff),
            money(enterprise_value),
            f"({money(debt)})",
            money(cash),
            money(equity_value),
            f"{shares_outstanding:,.0f}",
            money(intrinsic_value_per_share),
        ],
    }
)
st.dataframe(terminal_table, use_container_width=True, hide_index=True)

st.subheader("Terminal Value Check")
st.write("Terminal value uses the cash flow one year after the final forecast year.")

terminal_check = pd.DataFrame(
    {
        "Step": [
            "1. Grow final forecast FCFF one more year",
            "2. Apply the terminal value formula",
        ],
        "Calculation": [
            f"{money(final_fcff)} × (1 + {percent(terminal_growth)}) = {money(next_year_fcff)}",
            f"{money(next_year_fcff)} ÷ ({percent(wacc)} − {percent(terminal_growth)}) = {money(terminal_value)}",
        ],
    }
)
st.dataframe(terminal_check, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# DCF walkthrough
# ------------------------------------------------------------
st.header("DCF Walkthrough")

with st.expander("1. Start with recent revenue", expanded=True):
    st.write(
        "The model begins with the most recent annual revenue. For Celsius, this is useful because the company has had high sales growth, "
        "so the starting revenue level matters a lot."
    )
    st.latex(r"Revenue_0 = Most\ Recent\ Annual\ Revenue")

with st.expander("2. Forecast revenue"):
    st.write("Revenue is grown by the same selected growth rate during the forecast period.")
    st.latex(r"Revenue_t = Revenue_{t-1} \times (1 + Growth\ Rate)")

with st.expander("3. Estimate operating profit"):
    st.write("EBIT is estimated by applying the selected EBIT margin to forecast revenue.")
    st.latex(r"EBIT_t = Revenue_t \times EBIT\ Margin")

with st.expander("4. Calculate after-tax operating profit"):
    st.write("NOPAT removes taxes from EBIT before considering financing choices.")
    st.latex(r"NOPAT_t = EBIT_t \times (1 - Tax\ Rate)")

with st.expander("5. Estimate FCFF"):
    st.write(
        "This app uses a simple reinvestment assumption. A portion of NOPAT is reinvested, and the rest is treated as free cash flow to the firm."
    )
    st.latex(r"Reinvestment_t = NOPAT_t \times Reinvestment\ Rate")
    st.latex(r"FCFF_t = NOPAT_t - Reinvestment_t")

with st.expander("6. Build WACC from its components"):
    st.write(
        "Because FCFF belongs to both debt and equity investors, the discount rate is WACC. The cost of equity is estimated with CAPM, "
        "and the cost of debt is tax-adjusted."
    )
    st.latex(r"r_E = r_f + \beta \times ERP")
    st.latex(r"WACC = w_E r_E + w_D r_D(1 - Tax\ Rate)")

with st.expander("7. Discount forecast FCFF"):
    st.write("Each forecast cash flow is discounted back to present value using WACC.")
    st.latex(r"PV(FCFF_t) = \frac{FCFF_t}{(1 + WACC)^t}")

with st.expander("8. Calculate terminal value correctly"):
    st.write(
        "Terminal value is calculated at the end of the final forecast year using the following year's FCFF. "
        "That is why the final projected FCFF is multiplied by one plus the perpetual growth rate."
    )
    st.latex(r"Terminal\ Value_N = \frac{FCFF_N \times (1 + g)}{WACC - g}")
    st.latex(r"PV(Terminal\ Value) = \frac{Terminal\ Value_N}{(1 + WACC)^N}")

with st.expander("9. Convert enterprise value into value per share"):
    st.write(
        "Enterprise value is the value of the whole firm. To get equity value, subtract debt and add cash. "
        "Then divide by shares outstanding to calculate intrinsic value per share."
    )
    st.latex(r"Enterprise\ Value = \sum PV(FCFF_t) + PV(Terminal\ Value)")
    st.latex(r"Equity\ Value = Enterprise\ Value - Debt + Cash")
    st.latex(r"Intrinsic\ Value\ Per\ Share = \frac{Equity\ Value}{Shares\ Outstanding}")


# ------------------------------------------------------------
# Excel replication
# ------------------------------------------------------------
st.header("Excel Replication Table")
st.write(
    "This table is intentionally simple so the same valuation can be recreated in Excel with one row per forecast year."
)

excel_export = projection.copy()
excel_export["WACC"] = wacc
excel_export["Terminal Growth"] = terminal_growth
excel_export["Terminal Value"] = [np.nan] * (projection_years - 1) + [terminal_value]
excel_export["PV of Terminal Value"] = [np.nan] * (projection_years - 1) + [pv_terminal_value]
excel_export["Enterprise Value"] = [np.nan] * (projection_years - 1) + [enterprise_value]
excel_export["Debt"] = [np.nan] * (projection_years - 1) + [debt]
excel_export["Cash"] = [np.nan] * (projection_years - 1) + [cash]
excel_export["Equity Value"] = [np.nan] * (projection_years - 1) + [equity_value]
excel_export["Shares Outstanding"] = [np.nan] * (projection_years - 1) + [shares_outstanding]
excel_export["Intrinsic Value Per Share"] = [np.nan] * (projection_years - 1) + [intrinsic_value_per_share]

st.dataframe(excel_export, use_container_width=True, hide_index=True)

csv = excel_export.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV for Excel",
    data=csv,
    file_name=f"{ticker}_dcf_excel_replication.csv",
    mime="text/csv",
)

st.markdown(
    """
    **Suggested Excel layout:**
    1. Put assumptions at the top: reported revenue, growth, EBIT margin, tax rate, reinvestment rate, WACC inputs, terminal growth, reported debt, reported cash, and reported shares outstanding.
    2. Use one row for each forecast year.
    3. Link each formula to the assumption cells.
    4. Calculate terminal value only in the final forecast year.
    5. Add PV of FCFF and PV of terminal value, then subtract debt, add cash, and divide by shares outstanding.
    """
)
