import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsPulse",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&family=DM+Mono&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stApp"] {
    background: #0a0a0f;
    color: #e8e4dc;
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0e0e16 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] .stRadio label {
    color: #9090a8 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 6px 0;
    transition: color 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #e8c97e !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }

/* Sidebar logo area */
.sidebar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: #e8c97e;
    letter-spacing: -0.02em;
    padding: 0 1rem 1.5rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 1.5rem;
}
.sidebar-brand span { color: #e8e4dc; }

/* ── MAIN CONTENT ── */
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px;
}

/* ── PAGE HEADER ── */
.page-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid #1e1e2e;
    padding-bottom: 1.5rem;
}
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 900;
    color: #e8e4dc;
    line-height: 1;
    margin: 0;
}
.page-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #e8c97e;
    background: rgba(232,201,126,0.1);
    border: 1px solid rgba(232,201,126,0.25);
    padding: 3px 10px;
    border-radius: 2px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── METRIC CARDS ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #0e0e16;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.total::before  { background: #e8c97e; }
.metric-card.pos::before    { background: #4ade80; }
.metric-card.neu::before    { background: #60a5fa; }
.metric-card.neg::before    { background: #f87171; }
.metric-card:hover          { border-color: #2e2e3e; }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a5a72;
    margin-bottom: 0.75rem;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: #e8e4dc;
    line-height: 1;
}

/* ── SECTION LABELS ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a5a72;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e1e2e;
}

/* ── DATA TABLE ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e1e2e !important;
    border-radius: 4px;
}
[data-testid="stDataFrame"] th {
    background: #0e0e16 !important;
    color: #9090a8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── INPUTS ── */
.stTextInput input {
    background: #0e0e16 !important;
    border: 1px solid #2e2e3e !important;
    border-radius: 4px !important;
    color: #e8e4dc !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #e8c97e !important;
    box-shadow: 0 0 0 2px rgba(232,201,126,0.1) !important;
}

/* ── BUTTONS ── */
.stButton button {
    background: transparent !important;
    border: 1px solid #2e2e3e !important;
    color: #9090a8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 3px !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    border-color: #e8c97e !important;
    color: #e8c97e !important;
    background: rgba(232,201,126,0.05) !important;
}

/* ── LOGIN CARD ── */
.login-wrap {
    max-width: 420px;
    margin: 5vh auto 0;
    background: #0e0e16;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 3rem 2.5rem;
}
.login-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e8e4dc;
    margin-bottom: 0.25rem;
}
.login-subtitle {
    font-size: 0.82rem;
    color: #5a5a72;
    margin-bottom: 2rem;
}

/* ── HOME PAGE ── */
.home-hero {
    text-align: center;
    padding: 8vh 0 4vh;
}
.home-logo {
    font-family: 'Playfair Display', serif;
    font-size: 5rem;
    font-weight: 900;
    color: #e8e4dc;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.home-logo em { color: #e8c97e; font-style: normal; }
.home-tagline {
    font-size: 0.9rem;
    color: #5a5a72;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 4rem;
}
.role-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    max-width: 600px;
    margin: 0 auto;
}
.role-card {
    background: #0e0e16;
    border: 1px solid #1e1e2e;
    border-radius: 6px;
    padding: 2rem;
    cursor: pointer;
    transition: all 0.2s;
    text-align: left;
}
.role-card:hover { border-color: #e8c97e; }
.role-icon { font-size: 1.8rem; margin-bottom: 1rem; }
.role-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #e8e4dc;
    margin-bottom: 0.4rem;
}
.role-desc { font-size: 0.78rem; color: #5a5a72; }

/* ── SEARCH RESULT CARDS ── */
.news-card {
    background: #0e0e16;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #e8c97e;
    border-radius: 3px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-left-color 0.2s;
}
.news-card:hover { border-left-color: #f0d98e; }
.news-headline {
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    color: #e8e4dc;
    margin-bottom: 0.5rem;
    line-height: 1.4;
}
.news-link a {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #e8c97e;
    text-decoration: none;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── TOPIC PILL ── */
.topic-block {
    background: #0e0e16;
    border: 1px solid #1e1e2e;
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.topic-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #e8c97e;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.topic-words {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.topic-word {
    background: rgba(232,201,126,0.07);
    border: 1px solid rgba(232,201,126,0.15);
    color: #e8e4dc;
    font-size: 0.8rem;
    padding: 3px 10px;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
}

/* ── PLOTLY THEME FIX ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── USER MANAGEMENT ── */
.user-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
}
.user-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a5a72;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #1e1e2e;
    text-align: left;
}
.user-table td {
    padding: 0.85rem 1rem;
    border-bottom: 1px solid #12121a;
    color: #c8c4bc;
    vertical-align: middle;
}
.user-table tr:hover td { background: #0e0e16; }
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    display: inline-block;
}
.badge-active  { color: #4ade80; background: rgba(74,222,128,0.1);  border: 1px solid rgba(74,222,128,0.25); }
.badge-idle    { color: #e8c97e; background: rgba(232,201,126,0.1); border: 1px solid rgba(232,201,126,0.25);}
.badge-offline { color: #f87171; background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.25);}
.badge-admin   { color: #a78bfa; background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.25);}
.badge-user    { color: #60a5fa; background: rgba(96,165,250,0.1);  border: 1px solid rgba(96,165,250,0.25); }
.user-avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1e1e2e, #2e2e3e);
    display: inline-flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 0.85rem;
    color: #e8c97e;
    border: 1px solid #2e2e3e;
    margin-right: 10px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("news_with_sentiment.csv")
    df = df.dropna()
    df = df.drop_duplicates(subset=["processed_text"])
    return df

df = load_data()

def impact(sentiment):
    if sentiment == "Positive":   return "High"
    elif sentiment == "Neutral":  return "Medium"
    else:                         return "Low"

df["impact_level"] = df["sentiment"].apply(impact)

# ── CHART DEFAULTS ─────────────────────────────────────────────────────────────
CHART_BG    = "rgba(0,0,0,0)"
GRID_COLOR  = "#1e1e2e"
TEXT_COLOR  = "#9090a8"
ACCENT      = "#e8c97e"
COLOR_SEQ   = ["#e8c97e", "#60a5fa", "#4ade80", "#f87171", "#a78bfa", "#fb923c"]

def styled_layout(fig, title=""):
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="DM Sans", color=TEXT_COLOR, size=12),
        title=dict(text=title, font=dict(family="Playfair Display", size=18, color="#e8e4dc")) if title else {},
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickcolor=GRID_COLOR),
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR),
    )
    return fig

# ── HOME ───────────────────────────────────────────────────────────────────────
def home():
    st.markdown("""
    <div class="home-hero">
        <div class="home-logo">News<em>Pulse</em></div>
        <div class="home-tagline">Intelligence · Sentiment · Trends</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="section-label">Choose your access level</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⚙  Admin Portal", use_container_width=True):
                st.session_state.page = "admin_login"
                st.rerun()
        with c2:
            if st.button("◎  User Portal", use_container_width=True):
                st.session_state.page = "user_login"
                st.rerun()

# ── LOGIN ──────────────────────────────────────────────────────────────────────
def login_page(role: str):
    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        icon = "⚙" if role == "admin" else "◎"
        st.markdown(f"""
        <div class="login-title">{icon} &nbsp;{role.capitalize()} Login</div>
        <div class="login-subtitle">Enter your credentials to continue</div>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Sign In →", use_container_width=True):
                creds = {"admin": ("admin", "admin123"), "user": ("user", "user123")}
                valid_u, valid_p = creds[role]
                if username == valid_u and password == valid_p:
                    st.session_state.page = f"{role}_dashboard"
                    st.rerun()
                else:
                    st.error("Invalid credentials — please try again.")
        with c2:
            if st.button("← Back", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
def render_sidebar(role: str, items: list[str]) -> str:
    with st.sidebar:
        st.markdown(f'<div class="sidebar-brand">News<span>Pulse</span></div>', unsafe_allow_html=True)
        choice = st.radio("", items, label_visibility="collapsed")
        st.markdown("---")
        if st.button("Sign Out"):
            st.session_state.page = "home"
            st.rerun()
    return choice

# ── ADMIN DASHBOARD ────────────────────────────────────────────────────────────
def admin_dashboard():
    menu = render_sidebar("admin", [
        "Overview", "Dataset", "Trending Keywords",
        "Word Cloud", "Topic Modeling", "Sentiment Analysis",
        "Impact Level", "Source Analysis", "Search News",
        "User Management",
    ])

    st.markdown(f"""
    <div class="page-header">
        <h1 class="page-title">{menu}</h1>
        <span class="page-tag">Admin</span>
    </div>
    """, unsafe_allow_html=True)

    # ── OVERVIEW ──
    if menu == "Overview":
        total   = len(df)
        pos     = len(df[df["sentiment"] == "Positive"])
        neu     = len(df[df["sentiment"] == "Neutral"])
        neg     = len(df[df["sentiment"] == "Negative"])
        pos_pct = round(pos / total * 100, 1) if total else 0
        neg_pct = round(neg / total * 100, 1) if total else 0
        neu_pct = round(neu / total * 100, 1) if total else 0

        # ── top source & keyword
        top_source  = df["source_name"].value_counts().idxmax() if "source_name" in df.columns else "N/A"
        words       = " ".join(df["processed_text"]).split()
        top_keyword = Counter(words).most_common(1)[0][0] if words else "N/A"
        hi_impact   = len(df[df["impact_level"] == "High"])
        uniq_src    = df["source_name"].nunique() if "source_name" in df.columns else 0

        # ── ROW 1 — KPI metrics (8 cards) ──────────────────────────────────────
        st.markdown(f"""
        <div class="metric-grid" style="grid-template-columns:repeat(4,1fr)">
            <div class="metric-card total">
                <div class="metric-label">Total Articles</div>
                <div class="metric-value">{total:,}</div>
            </div>
            <div class="metric-card pos">
                <div class="metric-label">Positive</div>
                <div class="metric-value">{pos:,}</div>
                <div style="font-size:0.75rem;color:#4ade80;margin-top:4px">▲ {pos_pct}% of corpus</div>
            </div>
            <div class="metric-card neu">
                <div class="metric-label">Neutral</div>
                <div class="metric-value">{neu:,}</div>
                <div style="font-size:0.75rem;color:#60a5fa;margin-top:4px">◆ {neu_pct}%</div>
            </div>
            <div class="metric-card neg">
                <div class="metric-label">Negative</div>
                <div class="metric-value">{neg:,}</div>
                <div style="font-size:0.75rem;color:#f87171;margin-top:4px">▼ {neg_pct}%</div>
            </div>
        </div>
        <div class="metric-grid" style="grid-template-columns:repeat(4,1fr);margin-top:0">
            <div class="metric-card total">
                <div class="metric-label">High Impact</div>
                <div class="metric-value">{hi_impact:,}</div>
                <div style="font-size:0.75rem;color:#9090a8;margin-top:4px">articles flagged</div>
            </div>
            <div class="metric-card total">
                <div class="metric-label">Unique Sources</div>
                <div class="metric-value">{uniq_src}</div>
                <div style="font-size:0.75rem;color:#9090a8;margin-top:4px">publishers tracked</div>
            </div>
            <div class="metric-card total">
                <div class="metric-label">Top Source</div>
                <div class="metric-value" style="font-size:1.1rem;margin-top:6px">{top_source[:18]}</div>
            </div>
            <div class="metric-card total">
                <div class="metric-label">Top Keyword</div>
                <div class="metric-value" style="font-size:1.5rem;color:#e8c97e">{top_keyword}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW 2 — Sentiment donut + Impact bar ──────────────────────────────
        col1, col2 = st.columns(2)
        color_map = {"Positive": "#4ade80", "Neutral": "#60a5fa", "Negative": "#f87171"}
        impact_map = {"High": "#4ade80", "Medium": "#e8c97e", "Low": "#f87171"}

        with col1:
            st.markdown('<div class="section-label">Sentiment Share</div>', unsafe_allow_html=True)
            counts = df["sentiment"].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                hole=0.58,
                marker=dict(colors=[color_map.get(s, ACCENT) for s in counts.index],
                            line=dict(color="#0a0a0f", width=3)),
                textfont=dict(family="DM Mono", size=11),
            ))
            fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:10px'>articles</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="#e8e4dc", family="Playfair Display"))
            fig = styled_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Impact Level Distribution</div>', unsafe_allow_html=True)
            ic = df["impact_level"].value_counts()
            fig = go.Figure([go.Bar(
                x=ic.index, y=ic.values,
                marker_color=[impact_map.get(s, ACCENT) for s in ic.index],
                marker_line_width=0,
                text=ic.values, textposition="outside",
                textfont=dict(family="DM Mono", size=11, color="#9090a8"),
            )])
            fig = styled_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── ROW 3 — Top Sources bar + Top Keywords bar ─────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="section-label">Top 8 Sources</div>', unsafe_allow_html=True)
            if "source_name" in df.columns:
                src = df["source_name"].value_counts().head(8)
                fig = go.Figure([go.Bar(
                    x=src.values, y=src.index, orientation="h",
                    marker=dict(color=src.values, colorscale=[[0, "#1e1e2e"], [1, ACCENT]], line_width=0),
                    text=src.values, textposition="outside",
                    textfont=dict(family="DM Mono", size=10, color="#9090a8"),
                )])
                fig.update_layout(yaxis=dict(autorange="reversed"))
                fig = styled_layout(fig)
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown('<div class="section-label">Top 8 Keywords</div>', unsafe_allow_html=True)
            top8 = Counter(words).most_common(8)
            kw, kc = [i[0] for i in top8], [i[1] for i in top8]
            fig = go.Figure([go.Bar(
                x=kc, y=kw, orientation="h",
                marker=dict(color=kc, colorscale=[[0, "#1e1e2e"], [1, "#60a5fa"]], line_width=0),
                text=kc, textposition="outside",
                textfont=dict(family="DM Mono", size=10, color="#9090a8"),
            )])
            fig.update_layout(yaxis=dict(autorange="reversed"))
            fig = styled_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── ROW 4 — Recent articles feed ───────────────────────────────────────
        st.markdown('<div class="section-label">Latest Articles</div>', unsafe_allow_html=True)
        sent_badge_color = {"Positive": "#4ade80", "Neutral": "#60a5fa", "Negative": "#f87171"}
        for _, row in df.head(6).iterrows():
            scolor = sent_badge_color.get(row.get("sentiment", ""), ACCENT)
            badge  = f'<span style="font-family:DM Mono,monospace;font-size:0.62rem;color:{scolor};background:rgba(0,0,0,0.3);border:1px solid {scolor}33;padding:2px 8px;border-radius:2px;letter-spacing:0.1em;text-transform:uppercase">{row.get("sentiment","")}</span>'
            impact_badge = f'<span style="font-family:DM Mono,monospace;font-size:0.62rem;color:#9090a8;margin-left:6px;letter-spacing:0.08em">Impact: {row.get("impact_level","")}</span>'
            url_part = f'&nbsp;&nbsp;<a href="{row["url"]}" target="_blank" style="font-family:DM Mono,monospace;font-size:0.65rem;color:#e8c97e;text-decoration:none;letter-spacing:0.08em">Read →</a>' if "url" in row else ""
            st.markdown(f"""
            <div class="news-card" style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div class="news-headline" style="margin-bottom:6px">{row['title']}</div>
                    <div>{badge}{impact_badge}</div>
                </div>
                <div style="white-space:nowrap;margin-left:1rem">{url_part}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── DATASET ──
    elif menu == "Dataset":
        st.markdown('<div class="section-label">First 30 records</div>', unsafe_allow_html=True)
        st.dataframe(df.head(30), use_container_width=True, height=520)

    # ── TRENDING KEYWORDS ──
    elif menu == "Trending Keywords":
        words   = " ".join(df["processed_text"]).split()
        counter = Counter(words)
        top     = counter.most_common(15)
        w, c    = [i[0] for i in top], [i[1] for i in top]

        fig = go.Figure([go.Bar(
            x=c, y=w, orientation="h",
            marker=dict(
                color=c,
                colorscale=[[0, "#1e1e2e"], [1, ACCENT]],
                line_width=0,
            ),
        )])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig = styled_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── WORD CLOUD ──
    elif menu == "Word Cloud":
        st.markdown('<div class="section-label">Generated from processed text</div>', unsafe_allow_html=True)
        text = " ".join(df["processed_text"])
        wc   = WordCloud(
            width=1400, height=500,
            background_color="#0a0a0f",
            colormap="YlOrBr",
            max_words=120,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0a0a0f")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)

    # ── TOPIC MODELING ──
    elif menu == "Topic Modeling":
        st.markdown('<div class="section-label">LDA — 3 topics extracted via TF-IDF</div>', unsafe_allow_html=True)
        with st.spinner("Running topic model…"):
            vectorizer = TfidfVectorizer(max_features=1000)
            X = vectorizer.fit_transform(df["processed_text"])
            lda = LatentDirichletAllocation(n_components=3, random_state=42)
            lda.fit(X)
            vocab = vectorizer.get_feature_names_out()

        for i, topic in enumerate(lda.components_):
            top_idx   = topic.argsort()[-10:][::-1]
            top_words = [vocab[j] for j in top_idx]
            pills     = "".join(f'<span class="topic-word">{w}</span>' for w in top_words)
            st.markdown(f"""
            <div class="topic-block">
                <div class="topic-num">Topic {i + 1}</div>
                <div class="topic-words">{pills}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── SENTIMENT ANALYSIS ──
    elif menu == "Sentiment Analysis":
        counts = df["sentiment"].value_counts()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-label">Share</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                hole=0.55,
                marker=dict(colors=["#4ade80", "#60a5fa", "#f87171"],
                            line=dict(color="#0a0a0f", width=3)),
                textfont=dict(family="DM Mono", size=12),
            ))
            fig = styled_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Volume</div>', unsafe_allow_html=True)
            color_map = {"Positive": "#4ade80", "Neutral": "#60a5fa", "Negative": "#f87171"}
            fig = go.Figure([go.Bar(
                x=counts.index, y=counts.values,
                marker_color=[color_map.get(s, ACCENT) for s in counts.index],
                marker_line_width=0,
            )])
            fig = styled_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── IMPACT LEVEL ──
    elif menu == "Impact Level":
        counts = df["impact_level"].value_counts()
        color_map = {"High": "#4ade80", "Medium": "#e8c97e", "Low": "#f87171"}
        fig = go.Figure([go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[color_map.get(s, ACCENT) for s in counts.index],
            marker_line_width=0,
        )])
        fig = styled_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── SOURCE ANALYSIS ──
    elif menu == "Source Analysis":
        source = df["source_name"].value_counts().head(10)
        fig = go.Figure([go.Bar(
            x=source.values, y=source.index, orientation="h",
            marker=dict(color=source.values, colorscale=[[0, "#1e1e2e"], [1, ACCENT]], line_width=0),
        )])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig = styled_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── SEARCH ──
    elif menu == "Search News":
        keyword = st.text_input("", placeholder="Search by keyword in title…")
        if keyword:
            result = df[df["title"].str.contains(keyword, case=False, na=False)]
            st.markdown(f'<div class="section-label">{len(result)} results for "{keyword}"</div>', unsafe_allow_html=True)
            for _, row in result.head(10).iterrows():
                url_html = f'<div class="news-link"><a href="{row["url"]}" target="_blank">Read Article →</a></div>' \
                           if "url" in row else ""
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-headline">{row['title']}</div>
                    {url_html}
                </div>
                """, unsafe_allow_html=True)

# ── USER DASHBOARD ─────────────────────────────────────────────────────────────
def user_dashboard():
    menu = render_sidebar("user", [
        "Overview", "Trending Keywords", "Sentiment Analysis", "Search News",
    ])

    st.markdown(f"""
    <div class="page-header">
        <h1 class="page-title">{menu}</h1>
        <span class="page-tag">Reader</span>
    </div>
    """, unsafe_allow_html=True)

    if menu == "Overview":
        total   = len(df)
        pos     = len(df[df["sentiment"] == "Positive"])
        neu     = len(df[df["sentiment"] == "Neutral"])
        neg     = len(df[df["sentiment"] == "Negative"])
        pos_pct = round(pos / total * 100, 1) if total else 0
        neg_pct = round(neg / total * 100, 1) if total else 0

        words       = " ".join(df["processed_text"]).split()
        top_keyword = Counter(words).most_common(1)[0][0] if words else "N/A"
        top_source  = df["source_name"].value_counts().idxmax() if "source_name" in df.columns else "N/A"

        # ── KPI row ─────────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-grid" style="grid-template-columns:repeat(4,1fr)">
            <div class="metric-card total">
                <div class="metric-label">Total Articles</div>
                <div class="metric-value">{total:,}</div>
            </div>
            <div class="metric-card pos">
                <div class="metric-label">Positive</div>
                <div class="metric-value">{pos:,}</div>
                <div style="font-size:0.75rem;color:#4ade80;margin-top:4px">▲ {pos_pct}%</div>
            </div>
            <div class="metric-card neg">
                <div class="metric-label">Negative</div>
                <div class="metric-value">{neg:,}</div>
                <div style="font-size:0.75rem;color:#f87171;margin-top:4px">▼ {neg_pct}%</div>
            </div>
            <div class="metric-card neu">
                <div class="metric-label">Neutral</div>
                <div class="metric-value">{neu:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Donut + Top Keywords ────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        color_map = {"Positive": "#4ade80", "Neutral": "#60a5fa", "Negative": "#f87171"}

        with col1:
            st.markdown('<div class="section-label">Sentiment Breakdown</div>', unsafe_allow_html=True)
            counts = df["sentiment"].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                hole=0.58,
                marker=dict(colors=[color_map.get(s, ACCENT) for s in counts.index],
                            line=dict(color="#0a0a0f", width=3)),
                textfont=dict(family="DM Mono", size=11),
            ))
            fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:10px'>articles</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="#e8e4dc", family="Playfair Display"))
            fig = styled_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Top Keywords</div>', unsafe_allow_html=True)
            top8 = Counter(words).most_common(8)
            kw, kc = [i[0] for i in top8], [i[1] for i in top8]
            fig = go.Figure([go.Bar(
                x=kc, y=kw, orientation="h",
                marker=dict(color=kc, colorscale=[[0, "#1e1e2e"], [1, "#60a5fa"]], line_width=0),
                text=kc, textposition="outside",
                textfont=dict(family="DM Mono", size=10, color="#9090a8"),
            )])
            fig.update_layout(yaxis=dict(autorange="reversed"))
            fig = styled_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        # ── Quick facts row ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-grid" style="grid-template-columns:repeat(2,1fr)">
            <div class="metric-card total">
                <div class="metric-label">Trending Keyword</div>
                <div class="metric-value" style="font-size:1.6rem;color:#e8c97e">{top_keyword}</div>
            </div>
            <div class="metric-card total">
                <div class="metric-label">Most Active Source</div>
                <div class="metric-value" style="font-size:1.1rem;margin-top:8px">{top_source[:24]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Latest articles feed ───────────────────────────────────────────────
        st.markdown('<div class="section-label">Latest Articles</div>', unsafe_allow_html=True)
        sent_badge_color = {"Positive": "#4ade80", "Neutral": "#60a5fa", "Negative": "#f87171"}
        for _, row in df.head(5).iterrows():
            scolor = sent_badge_color.get(row.get("sentiment", ""), ACCENT)
            badge  = f'<span style="font-family:DM Mono,monospace;font-size:0.62rem;color:{scolor};background:rgba(0,0,0,0.3);border:1px solid {scolor}33;padding:2px 8px;border-radius:2px;letter-spacing:0.1em;text-transform:uppercase">{row.get("sentiment","")}</span>'
            url_part = f'&nbsp;&nbsp;<a href="{row["url"]}" target="_blank" style="font-family:DM Mono,monospace;font-size:0.65rem;color:#e8c97e;text-decoration:none;letter-spacing:0.08em">Read →</a>' if "url" in row else ""
            st.markdown(f"""
            <div class="news-card" style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div class="news-headline" style="margin-bottom:6px">{row['title']}</div>
                    <div>{badge}</div>
                </div>
                <div style="white-space:nowrap;margin-left:1rem">{url_part}</div>
            </div>
            """, unsafe_allow_html=True)

    elif menu == "Trending Keywords":
        words   = " ".join(df["processed_text"]).split()
        counter = Counter(words)
        top     = counter.most_common(10)
        w, c    = [i[0] for i in top], [i[1] for i in top]

        fig = go.Figure([go.Bar(
            x=c, y=w, orientation="h",
            marker=dict(color=c, colorscale=[[0, "#1e1e2e"], [1, ACCENT]], line_width=0),
        )])
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig = styled_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "Sentiment Analysis":
        counts    = df["sentiment"].value_counts()
        color_map = {"Positive": "#4ade80", "Neutral": "#60a5fa", "Negative": "#f87171"}
        fig = go.Figure([go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[color_map.get(s, ACCENT) for s in counts.index],
            marker_line_width=0,
        )])
        fig = styled_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "Search News":
        keyword = st.text_input("", placeholder="Search by keyword in title…")
        if keyword:
            result = df[df["title"].str.contains(keyword, case=False, na=False)]
            st.markdown(f'<div class="section-label">{len(result)} results for "{keyword}"</div>', unsafe_allow_html=True)
            for _, row in result.head(10).iterrows():
                url_html = f'<div class="news-link"><a href="{row["url"]}" target="_blank">Read Article →</a></div>' \
                           if "url" in row else ""
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-headline">{row['title']}</div>
                    {url_html}
                </div>
                """, unsafe_allow_html=True)

# ── ROUTER ─────────────────────────────────────────────────────────────────────
page = st.session_state.page
if   page == "home":            home()
elif page == "admin_login":     login_page("admin")
elif page == "user_login":      login_page("user")
elif page == "admin_dashboard": admin_dashboard()
elif page == "user_dashboard":  user_dashboard()