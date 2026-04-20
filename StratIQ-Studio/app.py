import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import plotly.express as px


# ── Page config ────────────────────────────────────────────────────────────────

_LOGO = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(
    page_title="StratIQ Studio",
    page_icon=str(_LOGO) if _LOGO.exists() else "📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Language initialisation (must happen before any widget render) ─────────────

if "lang" not in st.session_state:
    st.session_state["lang"] = "ar"

lang: str = st.session_state["lang"]


# ── Translation dictionary ─────────────────────────────────────────────────────

TRANSLATIONS: Dict[str, Dict[str, object]] = {
    "en": {
        # sidebar
        "lang_selector_label": "🌐 Language",
        "sidebar_caption": "Real-data market intelligence",
        "industry_label": "Industry / Sector",
        "industry_placeholder": "Fintech, HealthTech, EV, EdTech…",
        "region_label": "Target Region",
        "company_type_label": "Company Type",
        "company_types": ["Startup", "SME", "Large Enterprise"],
        "horizon_label": "Investment Horizon",
        "horizons": [
            "Short-term (1–2 years)",
            "Mid-term (3–5 years)",
            "Long-term (5+ years)",
        ],
        "run_button": "Generate Brief",
        "data_source_note": (
            "Data source: World Bank Indicators API. "
            "AI layer: optional Groq via st.secrets."
        ),
        # hero
        "hero_desc": (
            "Executive-grade market intelligence with a polished dashboard, "
            "real macroeconomic indicators, and an optional AI briefing layer. "
            "Start with a country, choose your industry, and generate a credible "
            "market-entry snapshot."
        ),
        # states
        "idle_info": "Choose a region and industry from the sidebar, then click **Generate Brief**.",
        "industry_warning": "Enter an industry first.",
        "spinner_text": "Pulling real indicators and building the brief…",
        "wb_error": "Couldn't load World Bank data right now. Please try again in a moment.",
        "fallback_note": "Running in fallback mode — live data unavailable.",
        # metrics
        "metric_gdp": "GDP",
        "metric_growth": "GDP Growth",
        "metric_population": "Population",
        "metric_score": "Opportunity Score",
        # data room
        "data_room_title": "Data Room",
        "chart_indicator_label": "Chart indicator",
        "no_data_warning": "No data returned for this indicator.",
        # executive brief
        "brief_title": "Executive Brief",
        "strengths_label": "### Strengths",
        "opportunities_label": "### Opportunities",
        "risks_label": "### Risks",
        # boardroom
        "boardroom_title": "Boardroom Snapshot",
        "gdp_growth_chart": "GDP Growth Rate (annual %)",
        "inflation_chart": "Inflation (consumer prices, annual %)",
        # footer
        "caption": (
            "Sources: World Bank Indicators API / World Development Indicators. "
            "AI brief is grounded on these fetched values."
        ),
    },
    "ar": {
        # sidebar
        "lang_selector_label": "🌐 اللغة",
        "sidebar_caption": "ذكاء السوق بالبيانات الفعلية",
        "industry_label": "القطاع / الصناعة",
        "industry_placeholder": "تقنية مالية، رعاية صحية، سيارات كهربائية…",
        "region_label": "المنطقة المستهدفة",
        "company_type_label": "نوع الشركة",
        "company_types": ["شركة ناشئة", "شركة صغيرة ومتوسطة", "شركة كبرى"],
        "horizon_label": "أفق الاستثمار",
        "horizons": [
            "قصير المدى (١–٢ سنوات)",
            "متوسط المدى (٣–٥ سنوات)",
            "طويل المدى (٥+ سنوات)",
        ],
        "run_button": "إنشاء التقرير",
        "data_source_note": "المصدر: واجهة برمجة البنك الدولي. الذكاء الاصطناعي: Groq اختياري.",
        # hero
        "hero_desc": (
            "ذكاء سوقي بمستوى تنفيذي مع لوحة تحكم احترافية ومؤشرات اقتصادية "
            "كلية حقيقية وطبقة تحليل ذكاء اصطناعي اختيارية. "
            "اختر دولة وقطاعك وأنشئ لمحة موثوقة لدخول السوق."
        ),
        # states
        "idle_info": "اختر المنطقة والقطاع من الشريط الجانبي، ثم انقر على **إنشاء التقرير**.",
        "industry_warning": "يرجى إدخال قطاع أولاً.",
        "spinner_text": "جارٍ استرجاع المؤشرات الفعلية وإعداد التقرير…",
        "wb_error": "تعذّر تحميل بيانات البنك الدولي حالياً. يرجى المحاولة لاحقاً.",
        "fallback_note": "التطبيق يعمل في الوضع الاحتياطي — البيانات الفعلية غير متاحة.",
        # metrics
        "metric_gdp": "الناتج المحلي الإجمالي",
        "metric_growth": "نمو الناتج المحلي",
        "metric_population": "عدد السكان",
        "metric_score": "مؤشر الفرصة",
        # data room
        "data_room_title": "غرفة البيانات",
        "chart_indicator_label": "مؤشر الرسم البياني",
        "no_data_warning": "لا توجد بيانات لهذا المؤشر.",
        # executive brief
        "brief_title": "التقرير التنفيذي",
        "strengths_label": "### نقاط القوة",
        "opportunities_label": "### الفرص",
        "risks_label": "### المخاطر",
        # boardroom
        "boardroom_title": "لمحة مجلس الإدارة",
        "gdp_growth_chart": "معدل نمو الناتج المحلي (% سنوياً)",
        "inflation_chart": "التضخم (أسعار المستهلك، % سنوياً)",
        # footer
        "caption": (
            "المصادر: واجهة برمجة مؤشرات البنك الدولي / مؤشرات التنمية العالمية. "
            "التقرير التنفيذي مبني على البيانات المسترجعة."
        ),
    },
}

T = TRANSLATIONS[lang]

# Internal (English) values kept for API calls regardless of display language
_COMPANY_TYPES_EN = ["Startup", "SME", "Large Enterprise"]
_HORIZONS_EN = [
    "Short-term (1–2 years)",
    "Mid-term (3–5 years)",
    "Long-term (5+ years)",
]


# ── Base CSS ───────────────────────────────────────────────────────────────────

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
    .logo-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }
    .small-note {
        color: #9FB0D4;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── RTL overlay (injected only when Arabic is active) ─────────────────────────

if lang == "ar":
    st.markdown(
        """
        <style>
        /* Direction */
        .stApp, .block-container {
            direction: rtl;
        }
        [data-testid="stSidebar"] > div:first-child {
            direction: rtl;
            text-align: right;
        }
        /* Text alignment */
        .stMarkdown, .stMarkdown p, label,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] {
            text-align: right !important;
        }
        .hero p, .hero h1 {
            text-align: right;
        }
        /* Flip selectbox and input padding */
        [data-baseweb="select"] {
            direction: rtl;
        }
        [data-baseweb="input"] input {
            direction: rtl;
            text-align: right;
        }
        /* Table cells */
        [data-testid="stDataFrame"] td,
        [data-testid="stDataFrame"] th {
            text-align: right !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ── Data constants ─────────────────────────────────────────────────────────────

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


# ── Scoring & formatting ───────────────────────────────────────────────────────

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


# ── World Bank fetching ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_indicator_series(
    country_code: str,
    indicator_code: str,
    start_year: int = 2016,
    end_year: int = 2025,
) -> pd.DataFrame:
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


# ── AI brief ──────────────────────────────────────────────────────────────────

_FALLBACK_BRIEF = {
    "score": 50,
    "summary": "The live data source is temporarily unavailable. The app is running in fallback mode.",
    "strengths": ["App deployed successfully", "Fallback mode is active", "Core interface is available"],
    "opportunities": ["Reconnect live data", "Add cached snapshots", "Improve resilience"],
    "risks": ["External API timeout", "Data source instability", "Cold-start delays"],
    "sources": ["Fallback mode"],
}


@st.cache_data(show_spinner=False, ttl=1800)
def build_ai_brief(
    industry: str,
    region: str,
    company_type: str,
    horizon: str,
    snapshot: Dict[str, pd.DataFrame],
) -> Dict:
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


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo
    if _LOGO.exists():
        st.image(str(_LOGO), width=140)
    else:
        st.title("StratIQ Studio")

    st.caption(T["sidebar_caption"])

    st.divider()

    # Language selector — changing triggers a rerun so RTL CSS updates immediately
    lang_display = st.selectbox(
        T["lang_selector_label"],
        options=["العربية", "English"],
        index=0 if lang == "ar" else 1,
        key="_lang_select",
    )
    new_lang = "ar" if lang_display == "العربية" else "en"
    if new_lang != lang:
        st.session_state["lang"] = new_lang
        st.rerun()

    st.divider()

    industry = st.text_input(T["industry_label"], placeholder=T["industry_placeholder"])
    region = st.selectbox(T["region_label"], list(COUNTRY_MAP.keys()), index=0)

    company_type_display = st.selectbox(
        T["company_type_label"], T["company_types"]  # type: ignore[arg-type]
    )
    company_type = _COMPANY_TYPES_EN[T["company_types"].index(company_type_display)]  # type: ignore[index]

    horizon_display = st.selectbox(
        T["horizon_label"], T["horizons"]  # type: ignore[arg-type]
    )
    horizon = _HORIZONS_EN[T["horizons"].index(horizon_display)]  # type: ignore[index]

    run = st.button(T["run_button"], type="primary", use_container_width=True)

    st.markdown(
        f"<div class='small-note'>{T['data_source_note']}</div>",
        unsafe_allow_html=True,
    )


# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="hero">
      <h1>StratIQ Studio</h1>
      <p>{T["hero_desc"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Gate checks ────────────────────────────────────────────────────────────────

if not run:
    st.info(T["idle_info"])
    st.stop()

if not industry.strip():
    st.warning(T["industry_warning"])
    st.stop()

country_code = COUNTRY_MAP[region]


# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner(T["spinner_text"]):
    try:
        snapshot = fetch_market_snapshot(country_code)
    except Exception:
        snapshot = {name: pd.DataFrame(columns=["year", "value"]) for name in INDICATORS}

    try:
        brief = build_ai_brief(industry, region, company_type, horizon, snapshot)
    except Exception:
        brief = _FALLBACK_BRIEF


# ── Metric row ─────────────────────────────────────────────────────────────────

gdp = latest_value(snapshot["GDP (current US$)"])
growth = latest_value(snapshot["GDP growth (annual %)"])
population = latest_value(snapshot["Population, total"])
inflation = latest_value(snapshot["Inflation, consumer prices (annual %)"])

if all(df.empty for df in snapshot.values()):
    st.error(T["wb_error"])
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric(T["metric_gdp"], format_number(gdp, "usd"))
m2.metric(T["metric_growth"], format_number(growth, "percent"))
m3.metric(T["metric_population"], format_number(population, "population"))
m4.metric(T["metric_score"], f"{brief['score']}/100")


# ── Main layout ────────────────────────────────────────────────────────────────

left, right = st.columns([1.45, 1], gap="large")

with left:
    st.subheader(T["data_room_title"])
    indicator_name = st.selectbox(T["chart_indicator_label"], list(INDICATORS.keys()))
    chart_df = snapshot[indicator_name].copy()
    if chart_df.empty:
        st.warning(T["no_data_warning"])
    else:
        fig = px.line(chart_df, x="year", y="value", markers=True, title=indicator_name)
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

    table_rows = [
        {"Indicator": name, "Latest value": latest_value(df)}
        for name, df in snapshot.items()
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

with right:
    st.subheader(T["brief_title"])
    st.write(brief["summary"])

    st.markdown(T["strengths_label"])
    for item in brief.get("strengths", []):
        st.markdown(f"- {item}")

    st.markdown(T["opportunities_label"])
    for item in brief.get("opportunities", []):
        st.markdown(f"- {item}")

    st.markdown(T["risks_label"])
    for item in brief.get("risks", []):
        st.markdown(f"- {item}")


# ── Boardroom Snapshot ─────────────────────────────────────────────────────────

st.subheader(T["boardroom_title"])
c1, c2 = st.columns(2)
with c1:
    growth_df = snapshot["GDP growth (annual %)"]
    if not growth_df.empty:
        fig2 = px.bar(growth_df, x="year", y="value", title=T["gdp_growth_chart"])
        fig2.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig2, use_container_width=True)
with c2:
    inflation_df = snapshot["Inflation, consumer prices (annual %)"]
    if not inflation_df.empty:
        fig3 = px.area(inflation_df, x="year", y="value", title=T["inflation_chart"])
        fig3.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig3, use_container_width=True)

st.caption(T["caption"])
