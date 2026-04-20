import json
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


st.set_page_config(
    page_title="StratIQ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #070c18 0%, #0c1422 55%, #09101e 100%);
        color: #E8F0FF;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    div[data-testid="stMetric"] {
        background: rgba(13, 20, 38, 0.92);
        border: 1px solid rgba(99, 149, 255, 0.18);
        border-radius: 16px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #7A96C8 !important;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #EBF2FF !important;
        font-size: 1.55rem;
        font-weight: 700;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(37,99,235,0.14) 0%, rgba(12,20,34,0.97) 100%);
        border: 1px solid rgba(99,149,255,0.18);
        border-radius: 24px;
        padding: 32px 36px;
        margin-bottom: 22px;
    }
    .hero-card h1 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #F0F6FF;
        margin: 0 0 8px 0;
        letter-spacing: -0.03em;
    }
    .hero-card p {
        color: #7A96C8;
        font-size: 0.94rem;
        margin: 0;
        max-width: 680px;
        line-height: 1.65;
    }
    .swot-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-top: 14px;
    }
    .swot-cell {
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 130px;
    }
    .swot-s { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.22); }
    .swot-w { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.22); }
    .swot-o { background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.22); }
    .swot-t { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.22); }
    .swot-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
    }
    .swot-s .swot-label { color: #4ade80; }
    .swot-w .swot-label { color: #f87171; }
    .swot-o .swot-label { color: #60a5fa; }
    .swot-t .swot-label { color: #fbbf24; }
    .swot-item { font-size: 0.84rem; color: #B8CCE8; padding: 2px 0; }
    .phase-card {
        background: rgba(37,99,235,0.07);
        border: 1px solid rgba(99,149,255,0.16);
        border-left: 3px solid #3b82f6;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .phase-title { font-size: 0.72rem; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 4px; }
    .phase-focus { font-size: 0.96rem; color: #E8F0FF; font-weight: 600; margin-bottom: 10px; }
    .phase-milestone { font-size: 0.84rem; color: #7A96C8; padding: 2px 0; }
    .verdict-card {
        background: linear-gradient(135deg, rgba(37,99,235,0.18) 0%, rgba(12,20,34,0.97) 100%);
        border: 1px solid rgba(99,149,255,0.28);
        border-radius: 18px;
        padding: 26px 30px;
        margin-bottom: 18px;
    }
    .verdict-badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 14px;
    }
    .badge-enter { background: rgba(34,197,94,0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.35); }
    .badge-defer { background: rgba(245,158,11,0.18); color: #fbbf24; border: 1px solid rgba(245,158,11,0.35); }
    .badge-avoid { background: rgba(239,68,68,0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.35); }
    .driver-pill {
        display: inline-block;
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(59,130,246,0.22);
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 0.85rem;
        color: #93C5FD;
        margin: 4px 4px 4px 0;
    }
    .small-note { color: #3D5275; font-size: 0.80rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

REGIONS = [
    "Saudi Arabia", "United Arab Emirates", "Qatar", "Kuwait",
    "Bahrain", "Oman", "Egypt", "Jordan", "Morocco",
    "United States", "United Kingdom", "Germany", "India", "China", "Global",
]


# ── Groq analysis generator ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=1800)
def generate_analysis(
    industry: str, region: str, company_type: str, horizon: str, api_key: str
) -> Dict:
    prompt = f"""
You are a senior strategy consultant at a top-tier consulting firm.
Produce a rigorous, data-grounded market entry analysis in JSON.
Use ranges and qualifiers where precise data is unavailable — do not fabricate specific statistics.

Inputs:
- Industry / Sector: {industry}
- Target Market: {region}
- Company Type: {company_type}
- Investment Horizon: {horizon}

Return ONLY valid JSON with this exact structure:
{{
  "opportunity_score": <integer 0-100>,
  "market_size_current": "<e.g. $4.2B>",
  "market_size_projected": "<e.g. $9.8B by 2030>",
  "cagr": "<e.g. 11.4%>",
  "key_drivers": ["<driver 1>", "<driver 2>", "<driver 3>", "<driver 4>"],
  "swot": {{
    "strengths": ["<s1>", "<s2>", "<s3>"],
    "weaknesses": ["<w1>", "<w2>", "<w3>"],
    "opportunities": ["<o1>", "<o2>", "<o3>"],
    "threats": ["<t1>", "<t2>", "<t3>"]
  }},
  "competitors": [
    {{"name": "<name>", "market_share": "<e.g. 18%>", "strength": "<one line>", "weakness": "<one line>"}},
    {{"name": "<name>", "market_share": "<e.g. 14%>", "strength": "<one line>", "weakness": "<one line>"}},
    {{"name": "<name>", "market_share": "<e.g. 11%>", "strength": "<one line>", "weakness": "<one line>"}},
    {{"name": "<name>", "market_share": "<e.g. 8%>",  "strength": "<one line>", "weakness": "<one line>"}}
  ],
  "timeline": [
    {{"phase": "Phase 1 (0–6 months)",  "focus": "<focus>", "milestones": ["<m1>", "<m2>", "<m3>"]}},
    {{"phase": "Phase 2 (6–18 months)", "focus": "<focus>", "milestones": ["<m1>", "<m2>", "<m3>"]}},
    {{"phase": "Phase 3 (18+ months)",  "focus": "<focus>", "milestones": ["<m1>", "<m2>", "<m3>"]}}
  ],
  "risks": [
    {{"name": "<risk>", "likelihood": <1-10>, "impact": <1-10>, "mitigation": "<one line>"}},
    {{"name": "<risk>", "likelihood": <1-10>, "impact": <1-10>, "mitigation": "<one line>"}},
    {{"name": "<risk>", "likelihood": <1-10>, "impact": <1-10>, "mitigation": "<one line>"}},
    {{"name": "<risk>", "likelihood": <1-10>, "impact": <1-10>, "mitigation": "<one line>"}},
    {{"name": "<risk>", "likelihood": <1-10>, "impact": <1-10>, "mitigation": "<one line>"}}
  ],
  "verdict": "<3-5 sentence executive recommendation>",
  "recommendation": "<one of: Enter / Defer / Avoid>"
}}
"""
    try:
        resp = requests.post(
            GROQ_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1800,
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as exc:
        st.error(f"Analysis generation failed: {exc}")
        return {}


# ── Rendering helpers ──────────────────────────────────────────────────────────

def render_swot(swot: Dict) -> None:
    def items_html(lst: List[str]) -> str:
        return "".join(f'<div class="swot-item">· {i}</div>' for i in lst)

    st.markdown(
        f"""
        <div class="swot-grid">
          <div class="swot-cell swot-s">
            <div class="swot-label">Strengths</div>
            {items_html(swot.get("strengths", []))}
          </div>
          <div class="swot-cell swot-w">
            <div class="swot-label">Weaknesses</div>
            {items_html(swot.get("weaknesses", []))}
          </div>
          <div class="swot-cell swot-o">
            <div class="swot-label">Opportunities</div>
            {items_html(swot.get("opportunities", []))}
          </div>
          <div class="swot-cell swot-t">
            <div class="swot-label">Threats</div>
            {items_html(swot.get("threats", []))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_matrix(risks: List[Dict]) -> None:
    df = pd.DataFrame(risks)
    df["severity"] = df["likelihood"] * df["impact"]
    df["color"] = df["severity"].apply(
        lambda s: "#f87171" if s >= 60 else ("#fbbf24" if s >= 30 else "#4ade80")
    )

    fig = go.Figure()

    # background zone shading
    fig.add_shape(type="rect", x0=0, y0=0, x1=3.5, y1=10.5,
                  fillcolor="rgba(34,197,94,0.05)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=3.5, y0=0, x1=6.5, y1=10.5,
                  fillcolor="rgba(245,158,11,0.05)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=6.5, y0=0, x1=10.5, y1=10.5,
                  fillcolor="rgba(239,68,68,0.05)", line_width=0, layer="below")

    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["likelihood"]],
            y=[row["impact"]],
            mode="markers+text",
            text=[row["name"]],
            textposition="top center",
            textfont=dict(size=10.5, color="#B8CCE8"),
            marker=dict(
                size=20,
                color=row["color"],
                opacity=0.82,
                line=dict(width=1.5, color="rgba(255,255,255,0.2)"),
            ),
            hovertemplate=(
                f"<b>{row['name']}</b><br>"
                f"Likelihood: {row['likelihood']}/10<br>"
                f"Impact: {row['impact']}/10<br>"
                f"Mitigation: {row['mitigation']}<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,16,30,0.7)",
        height=390,
        xaxis=dict(
            title="Likelihood →",
            range=[0, 11],
            showgrid=True,
            gridcolor="rgba(99,149,255,0.08)",
            tickfont=dict(color="#7A96C8"),
        ),
        yaxis=dict(
            title="Impact →",
            range=[0, 11],
            showgrid=True,
            gridcolor="rgba(99,149,255,0.08)",
            tickfont=dict(color="#7A96C8"),
        ),
        margin=dict(l=40, r=20, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ StratIQ")
    st.caption("AI-powered market intelligence")
    st.divider()
    industry = st.text_input("Industry / Sector", placeholder="e.g. Fintech, HealthTech, EV...")
    region = st.selectbox("Target Market", REGIONS)
    company_type = st.selectbox("Company Type", ["Startup", "SME", "Large Enterprise", "PE / VC"])
    horizon = st.selectbox(
        "Investment Horizon",
        ["Short-term (0–2 years)", "Mid-term (2–5 years)", "Long-term (5+ years)"],
    )
    st.divider()
    run = st.button("Generate Analysis", type="primary", use_container_width=True)
    st.markdown(
        "<div class='small-note'>Powered by Groq · LLaMA 3.3 70B</div>",
        unsafe_allow_html=True,
    )


# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero-card">
      <h1>⚡ StratIQ</h1>
      <p>Professional market intelligence, SWOT analysis, competitive landscape, and strategic
      recommendations — generated in seconds using frontier AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not run:
    st.info("Choose your industry and target market in the sidebar, then click **Generate Analysis**.")
    st.stop()

if not industry.strip():
    st.warning("Enter an industry or sector first.")
    st.stop()


# ── API key ────────────────────────────────────────────────────────────────────

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY not found. Add it to `.streamlit/secrets.toml` locally or via Streamlit Cloud secrets.")
    st.stop()


# ── Generate ───────────────────────────────────────────────────────────────────

with st.spinner("Generating analysis — this takes about 10 seconds..."):
    data = generate_analysis(industry, region, company_type, horizon, api_key)

if not data:
    st.error("Could not generate analysis. Check your API key and try again.")
    st.stop()


# ── Top metrics ────────────────────────────────────────────────────────────────

score = int(data.get("opportunity_score", 50))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Opportunity Score", f"{score} / 100")
c2.metric("Market Size (Now)", data.get("market_size_current", "N/A"))
c3.metric("Projected Size", data.get("market_size_projected", "N/A"))
c4.metric("CAGR", data.get("cagr", "N/A"))

st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "SWOT Analysis", "Competitors", "Timeline", "Risk Matrix"]
)

# ── Tab 1: Overview ────────────────────────────────────────────────────────────
with tab1:
    rec = data.get("recommendation", "Defer")
    badge_class = {"Enter": "badge-enter", "Avoid": "badge-avoid"}.get(rec, "badge-defer")

    st.markdown(
        f"""
        <div class="verdict-card">
          <div class="verdict-badge {badge_class}">{rec}</div>
          <p style="color:#C0D4EE; font-size:0.96rem; line-height:1.75; margin:0;">
            {data.get("verdict", "")}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Key Market Drivers")
    drivers_html = "".join(
        f'<span class="driver-pill">{d}</span>' for d in data.get("key_drivers", [])
    )
    st.markdown(drivers_html, unsafe_allow_html=True)

# ── Tab 2: SWOT ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("SWOT Analysis")
    render_swot(data.get("swot", {}))

# ── Tab 3: Competitors ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Competitive Landscape")
    competitors = data.get("competitors", [])
    if competitors:
        comp_df = pd.DataFrame(competitors)
        comp_df.columns = ["Company", "Est. Market Share", "Core Strength", "Key Weakness"]
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No competitor data returned.")

# ── Tab 4: Timeline ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Market Entry Timeline")
    for phase in data.get("timeline", []):
        milestones_html = "".join(
            f'<div class="phase-milestone">· {m}</div>'
            for m in phase.get("milestones", [])
        )
        st.markdown(
            f"""
            <div class="phase-card">
              <div class="phase-title">{phase.get("phase", "")}</div>
              <div class="phase-focus">{phase.get("focus", "")}</div>
              {milestones_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Tab 5: Risk Matrix ─────────────────────────────────────────────────────────
with tab5:
    st.subheader("Risk Matrix")
    risks = data.get("risks", [])
    if risks:
        render_risk_matrix(risks)
        st.markdown("#### Risk Register")
        risk_df = pd.DataFrame(risks)[["name", "likelihood", "impact", "mitigation"]]
        risk_df.columns = ["Risk", "Likelihood (1–10)", "Impact (1–10)", "Mitigation"]
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
    else:
        st.info("No risk data returned.")


st.caption(
    f"StratIQ · {industry} in {region} · {company_type} · {horizon} · "
    "Powered by Groq LLaMA 3.3 70B · Data is AI-generated and for strategic guidance only."
)
