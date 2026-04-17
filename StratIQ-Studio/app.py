import json
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="StratIQ Studio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0B1020 0%, #11182c 100%);
        color: #EAF1FF;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }
    div[data-testid="stMetric"] {
        background: rgba(20, 27, 45, 0.92);
        border: 1px solid rgba(77, 124, 254, 0.18);
        border-radius: 18px;
        padding: 14px 18px;
    }
    .hero {
        background: linear-gradient(135deg, rgba(47,107,255,0.18), rgba(20,27,45,0.92));
        border: 1px solid rgba(77, 124, 254, 0.20);
        border-radius: 26px;
        padding: 28px;
        margin-bottom: 18px;
    }
    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.3rem;
        color: #F7FAFF;
    }
    .hero p {
        color: #B9C6E3;
        font-size: 1rem;
        max-width: 850px;
    }
    .small-note {
        color: #9FB0D4;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COUNTRY_MAP = {
    "Saudi Arabia": "SAU",
    "United Arab Emirates": "ARE",
    "Qatar": "QAT",
    "Kuwait": "KWT",
    "Bahrain": "BHR",
    "Oman": "OMN",
    "Egypt": "EGY",
    "Jordan": "JOR",
    "Morocco": "MAR",
    "Global": "WLD",
}

INDICATORS = {
    "GDP (current US$)": "NY.GDP.MKTP.CD",
    "GDP growth (annual %)": "NY.GDP.MKTP.KD.ZG",
    "Population, total": "SP.POP.TOTL",
    "Inflation, consumer prices (annual %)": "FP.CPI.TOTL.ZG",
}


def score_opportunity(gdp_growth: Optional[float], inflation: Optional[float]) -> int:
    score = 50
    if gdp_growth is not None:
        if gdp_growth >= 5:
            score += 15
        elif gdp_growth >= 2:
            score += 8
        elif gdp_growth < 0:
            score -= 10
    if inflation is not None:
        if inflation > 8:
            score -= 12
        elif inflation > 4:
            score -= 5
    return max(0, min(100, score))


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_indicator_series(country_code: str, indicator_code: str, start_year: int = 2016, end_year: int = 2025) -> pd.DataFrame:
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
        f"?format=json&per_page=200&date={start_year}:{end_year}"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame(columns=["year", "value"])

        rows = []
        for item in payload[1]:
            value = item.get("value")
            date = item.get("date")
            if value is not None and date is not None:
                rows.append({"year": int(date), "value": float(value)})

        if not rows:
            return pd.DataFrame(columns=["year", "value"])

        return pd.DataFrame(rows).sort_values("year")

    except Exception:
        return pd.DataFrame(columns=["year", "value"])


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_market_snapshot(country_code: str) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for name, code in INDICATORS.items():
        try:
            result[name] = fetch_indicator_series(country_code, code)
        except Exception:
            result[name] = pd.DataFrame(columns=["year", "value"])
    return result


def latest_value(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    return float(df.iloc[-1]["value"])


def format_number(value: Optional[float], kind: str = "number") -> str:
    if value is None:
        return "N/A"
    if kind == "usd":
        if abs(value) >= 1_000_000_000_000:
            return f"${value / 1_000_000_000_000:.2f}T"
        if abs(value) >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        return f"${value:,.0f}"
    if kind == "percent":
        return f"{value:.1f}%"
    if kind == "population":
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        return f"{value:,.0f}"
    return f"{value:,.2f}"


@st.cache_data(show_spinner=False, ttl=1800)
def build_ai_brief(industry: str, region: str, company_type: str, horizon: str, snapshot: Dict[str, pd.DataFrame]) -> Dict:
    key_present = False
    try:
        key_present = bool(st.secrets.get("GROQ_API_KEY", ""))
    except Exception:
        key_present = False

    gdp = latest_value(snapshot["GDP (current US$)"])
    growth = latest_value(snapshot["GDP growth (annual %)"])
    pop = latest_value(snapshot["Population, total"])
    inflation = latest_value(snapshot["Inflation, consumer prices (annual %)"])
    score = score_opportunity(growth, inflation)

    if not key_present:
        return {
            "score": score,
            "summary": (
                f"This draft uses real World Bank macro indicators for {region} and a rules-based scoring layer. "
                f"Add GROQ_API_KEY in Streamlit secrets to generate a richer AI brief for the {industry} opportunity."
            ),
            "strengths": [
                f"Population scale: {format_number(pop, 'population')}",
                f"GDP size: {format_number(gdp, 'usd')}",
                f"GDP growth: {format_number(growth, 'percent')}",
            ],
            "risks": [
                f"Inflation environment: {format_number(inflation, 'percent')}",
                f"Macro volatility may affect {industry.lower()} demand.",
                f"Execution difficulty depends on regulation and local partnerships.",
            ],
            "opportunities": [
                f"GDP growth is currently {format_number(growth, 'percent')}.",
                f"Market scale supported by a population of {format_number(pop, 'population')}.",
                f"GDP base of {format_number(gdp, 'usd')} signals a sizable consumer market.",
            ],
            "sources": ["World Bank Indicators API"],
        }

    prompt = f"""
You are a senior strategy consultant.
Create a concise executive brief for market entry.
Use the facts below as ground truth and do not invent conflicting numbers.

Industry: {industry}
Region: {region}
Company Type: {company_type}
Investment Horizon: {horizon}

Facts:
- GDP (current US$): {gdp}
- GDP growth (annual %): {growth}
- Population: {pop}
- Inflation (annual %): {inflation}
- Opportunity score: {score}

Return valid JSON only with this structure:
{{
  "score": 0,
  "summary": "4-6 sentence executive brief",
  "strengths": ["...", "...", "..."],
  "risks": ["...", "...", "..."],
  "opportunities": ["...", "...", "..."],
  "sources": ["World Bank Indicators API"]
}}
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.25,
            "max_tokens": 900,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"].strip()
    cleaned = text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(cleaned)
    parsed["score"] = score if not parsed.get("score") else int(parsed["score"])
    return parsed


with st.sidebar:
    st.title("StratIQ Studio")
    st.caption("Real-data market intelligence")
    industry = st.text_input("Industry / Sector", placeholder="Fintech, HealthTech, EV, EdTech...")
    region = st.selectbox("Target Region", list(COUNTRY_MAP.keys()), index=0)
    company_type = st.selectbox("Company Type", ["Startup", "SME", "Large Enterprise"])
    horizon = st.selectbox("Investment Horizon", ["Short-term (1–2 years)", "Mid-term (3–5 years)", "Long-term (5+ years)"])
    run = st.button("Generate Brief", type="primary", use_container_width=True)
    st.markdown("<div class='small-note'>Data source: World Bank Indicators API. AI layer: optional Groq via st.secrets.</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <h1>StratIQ Studio</h1>
      <p>Executive-grade market intelligence with a polished dashboard, real macroeconomic indicators, and an optional AI briefing layer. Start with a country, choose your industry, and generate a credible market-entry snapshot instead of a purely invented analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not run:
    st.info("Choose a region and industry from the left, then click Generate Brief.")
    st.stop()

if not industry.strip():
    st.warning("Enter an industry first.")
    st.stop()

country_code = COUNTRY_MAP[region]

_FALLBACK_BRIEF = {
    "score": 50,
    "summary": "The live data source is temporarily unavailable. The app is running in fallback mode.",
    "strengths": ["App deployed successfully", "Fallback mode is active", "Core interface is available"],
    "opportunities": ["Reconnect live data", "Add cached snapshots", "Improve resilience"],
    "risks": ["External API timeout", "Data source instability", "Cold-start delays"],
    "sources": ["Fallback mode"],
}

with st.spinner("Pulling real indicators and building the brief..."):
    try:
        snapshot = fetch_market_snapshot(country_code)
    except Exception:
        snapshot = {name: pd.DataFrame(columns=["year", "value"]) for name in INDICATORS.keys()}

    try:
        brief = build_ai_brief(industry, region, company_type, horizon, snapshot)
    except Exception:
        brief = _FALLBACK_BRIEF


gdp = latest_value(snapshot["GDP (current US$)"])
growth = latest_value(snapshot["GDP growth (annual %)"])
population = latest_value(snapshot["Population, total"])
inflation = latest_value(snapshot["Inflation, consumer prices (annual %)"])

if all(df.empty for df in snapshot.values()):
    st.error("Couldn't load World Bank data right now. Please try again in a moment.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("GDP", format_number(gdp, "usd"))
m2.metric("GDP Growth", format_number(growth, "percent"))
m3.metric("Population", format_number(population, "population"))
m4.metric("Opportunity Score", f"{brief['score']}/100")

left, right = st.columns([1.45, 1], gap="large")

with left:
    st.subheader("Data Room")
    indicator_name = st.selectbox("Chart indicator", list(INDICATORS.keys()))
    chart_df = snapshot[indicator_name].copy()
    if chart_df.empty:
        st.warning("No data returned for this indicator.")
    else:
        fig = px.line(chart_df, x="year", y="value", markers=True, title=indicator_name)
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

    table_rows = []
    for name, df in snapshot.items():
        table_rows.append({"Indicator": name, "Latest value": latest_value(df)})
    table_df = pd.DataFrame(table_rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

with right:
    st.subheader("Executive Brief")
    st.write(brief["summary"])

    st.markdown("### Strengths")
    for item in brief.get("strengths", []):
        st.markdown(f"- {item}")

    st.markdown("### Opportunities")
    for item in brief.get("opportunities", []):
        st.markdown(f"- {item}")

    st.markdown("### Risks")
    for item in brief.get("risks", []):
        st.markdown(f"- {item}")

st.subheader("Boardroom Snapshot")
c1, c2 = st.columns(2)
with c1:
    growth_df = snapshot["GDP growth (annual %)"]
    if not growth_df.empty:
        fig2 = px.bar(growth_df, x="year", y="value", title="GDP Growth Rate (annual %)")
        fig2.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig2, use_container_width=True)
with c2:
    inflation_df = snapshot["Inflation, consumer prices (annual %)"]
    if not inflation_df.empty:
        fig3 = px.area(inflation_df, x="year", y="value", title="Inflation (consumer prices, annual %)")
        fig3.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig3, use_container_width=True)

st.caption("Sources: World Bank Indicators API / World Development Indicators. AI brief is grounded on these fetched values.")