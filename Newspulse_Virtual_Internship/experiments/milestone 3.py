import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsPulse",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ---- Root palette ---- */
:root {
    --bg:        #050810;
    --surface:   #0d1117;
    --card:      #111827;
    --border:    rgba(255,255,255,0.06);
    --accent:    #00e5ff;
    --accent2:   #ff4d6d;
    --accent3:   #a78bfa;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --positive:  #22d3a5;
    --neutral:   #facc15;
    --negative:  #f87171;
}

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 100% !important; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #050810 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, var(--accent), var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 1rem 0 0.25rem;
}
.sidebar-tagline {
    font-size: 0.7rem;
    font-weight: 300;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 2rem;
}
.sidebar-divider {
    height: 1px;
    background: var(--border);
    margin: 1.2rem 0;
}

/* ---- Nav pills ---- */
[data-testid="stRadio"] > div { gap: 0.3rem !important; }
[data-testid="stRadio"] label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 0.55rem 0.9rem !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
[data-testid="stRadio"] label:hover {
    background: rgba(0,229,255,0.06) !important;
    border-color: rgba(0,229,255,0.2) !important;
}
[data-testid="stRadio"] label[data-selected="true"],
[data-testid="stRadio"] label[aria-checked="true"] {
    background: rgba(0,229,255,0.1) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ---- Page hero header ---- */
.page-hero {
    margin-bottom: 2rem;
}
.page-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.8rem, 3vw, 2.8rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 0.4rem;
    background: linear-gradient(135deg, #fff 30%, var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.page-hero p {
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 300;
    margin: 0;
}

/* ---- Metric cards ---- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.total::before  { background: linear-gradient(90deg, var(--accent), var(--accent3)); }
.metric-card.pos::before    { background: var(--positive); }
.metric-card.neu::before    { background: var(--neutral); }
.metric-card.neg::before    { background: var(--negative); }

.metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(0,0,0,0.4); }

.metric-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
}
.metric-card.total .metric-value { color: var(--accent); }
.metric-card.pos   .metric-value { color: var(--positive); }
.metric-card.neu   .metric-value { color: var(--neutral); }
.metric-card.neg   .metric-value { color: var(--negative); }

.metric-icon {
    position: absolute;
    right: 1.2rem;
    top: 1.2rem;
    font-size: 1.4rem;
    opacity: 0.25;
}

/* ---- Section cards ---- */
.section-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.section-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #fff;
    margin: 0 0 1.2rem;
}

/* ---- Topic pill badges ---- */
.topic-container { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
.topic-pill {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}
.topic-pill.t0 { background: rgba(0,229,255,0.12); color: var(--accent); border: 1px solid rgba(0,229,255,0.25); }
.topic-pill.t1 { background: rgba(167,139,250,0.12); color: var(--accent3); border: 1px solid rgba(167,139,250,0.25); }
.topic-pill.t2 { background: rgba(255,77,109,0.12); color: var(--accent2); border: 1px solid rgba(255,77,109,0.25); }

.topic-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.topic-header.t0 { color: var(--accent); }
.topic-header.t1 { color: var(--accent3); }
.topic-header.t2 { color: var(--accent2); }

/* ---- Search bar ---- */
[data-testid="stTextInput"] input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,0.08) !important;
}

/* ---- Dataframe ---- */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }

/* ---- Stat band (overview bottom) ---- */
.stat-band {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.8rem;
}
.stat-dot {
    width: 8px; height: 8px; border-radius: 50%;
}

/* ---- Result badge ---- */
.result-badge {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.2);
    color: var(--accent);
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("news_with_sentiment.csv")
    def impact_level(s):
        return "High" if s == "Positive" else ("Medium" if s == "Neutral" else "Low")
    df["impact_level"] = df["sentiment"].apply(impact_level)
    return df

df = load_data()

# ── Plotly theme helper ───────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#94a3b8", size=12),
    title_font=dict(family="Syne", color="#e2e8f0", size=15, weight=700),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    margin=dict(l=10, r=10, t=40, b=10),
)
COLORS = ["#00e5ff", "#a78bfa", "#ff4d6d", "#22d3a5", "#facc15", "#fb923c"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ NewsPulse</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI-Powered News Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Dataset", "Trending Keywords", "Word Cloud",
         "Topic Modeling", "Sentiment Analysis", "Impact Level",
         "Source Analysis", "Search News"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.7rem; color:#334155; line-height:1.8;'>
        <span style='color:#475569;'>Articles</span> &nbsp;{len(df):,}<br>
        <span style='color:#475569;'>Sources</span> &nbsp;{df['source_name'].nunique()}<br>
        <span style='color:#475569;'>Sentiments</span> &nbsp;3 classes
    </div>
    """, unsafe_allow_html=True)

# ── Pages ─────────────────────────────────────────────────────────────────────

# ---- Overview ----
if page == "Overview":
    st.markdown("""
    <div class="page-hero">
        <h1>News Intelligence<br>at a Glance</h1>
        <p>Real-time snapshot of sentiment, reach and editorial landscape.</p>
    </div>
    """, unsafe_allow_html=True)

    pos = len(df[df["sentiment"] == "Positive"])
    neu = len(df[df["sentiment"] == "Neutral"])
    neg = len(df[df["sentiment"] == "Negative"])

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card total">
            <div class="metric-icon">📰</div>
            <div class="metric-label">Total Articles</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        <div class="metric-card pos">
            <div class="metric-icon">🟢</div>
            <div class="metric-label">Positive</div>
            <div class="metric-value">{pos:,}</div>
        </div>
        <div class="metric-card neu">
            <div class="metric-icon">🟡</div>
            <div class="metric-label">Neutral</div>
            <div class="metric-value">{neu:,}</div>
        </div>
        <div class="metric-card neg">
            <div class="metric-icon">🔴</div>
            <div class="metric-label">Negative</div>
            <div class="metric-value">{neg:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-card"><h3>Sentiment Distribution</h3>', unsafe_allow_html=True)
        sc = df["sentiment"].value_counts()
        clr_map = {"Positive": "#22d3a5", "Neutral": "#facc15", "Negative": "#f87171"}
        fig = go.Figure(go.Bar(
            x=sc.index, y=sc.values,
            marker_color=[clr_map.get(s, "#6b7280") for s in sc.index],
            marker_line_width=0,
        ))
        fig.update_layout(**PLOT_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card"><h3>Impact Breakdown</h3>', unsafe_allow_html=True)
        ic = df["impact_level"].value_counts()
        clr_imp = {"High": "#22d3a5", "Medium": "#facc15", "Low": "#f87171"}
        fig2 = go.Figure(go.Pie(
            labels=ic.index, values=ic.values,
            hole=0.62,
            marker=dict(colors=[clr_imp.get(i, "#6b7280") for i in ic.index],
                        line=dict(color="#111827", width=3)),
            textfont=dict(family="DM Sans", size=12, color="#e2e8f0"),
        ))
        fig2.update_layout(**PLOT_LAYOUT, showlegend=True,
                           legend=dict(font=dict(color="#94a3b8", size=11)))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---- Dataset ----
elif page == "Dataset":
    st.markdown('<div class="page-hero"><h1>Dataset Explorer</h1><p>Browse the raw corpus powering NewsPulse.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card"><h3>Preview — first 30 rows</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(30), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Trending Keywords ----
elif page == "Trending Keywords":
    st.markdown('<div class="page-hero"><h1>Trending Keywords</h1><p>The words dominating today\'s headlines.</p></div>', unsafe_allow_html=True)

    all_words = " ".join(df["processed_text"]).split()
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(15)
    words = [i[0] for i in top_words]
    counts = [i[1] for i in top_words]

    st.markdown('<div class="section-card"><h3>Top 15 Keywords by Frequency</h3>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=words, y=counts,
        marker=dict(
            color=counts,
            colorscale=[[0, "#0d1f3c"], [0.5, "#0077ff"], [1, "#00e5ff"]],
            line_width=0,
        ),
        text=counts,
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.update_layout(**PLOT_LAYOUT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Word Cloud ----
elif page == "Word Cloud":
    st.markdown('<div class="page-hero"><h1>Word Cloud</h1><p>Visual frequency map of all processed text.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card"><h3>Keyword Density</h3>', unsafe_allow_html=True)
    text = " ".join(df["processed_text"])
    wc = WordCloud(
        width=1200, height=500,
        background_color="#111827",
        colormap="cool",
        max_words=200,
        prefer_horizontal=0.85,
        relative_scaling=0.5,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Topic Modeling ----
elif page == "Topic Modeling":
    st.markdown('<div class="page-hero"><h1>Topic Modeling</h1><p>Latent Dirichlet Allocation reveals hidden themes.</p></div>', unsafe_allow_html=True)

    with st.spinner("Running LDA…"):
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf = vectorizer.fit_transform(df["processed_text"])
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(tfidf)
        feature_names = vectorizer.get_feature_names_out()

    topic_colors = ["t0", "t1", "t2"]
    topic_labels = ["Topic A", "Topic B", "Topic C"]

    cols = st.columns(3)
    for idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-12:][::-1]
        topic_words = [feature_names[i] for i in top_indices]
        tc = topic_colors[idx]
        with cols[idx]:
            pills = "".join(f'<span class="topic-pill {tc}">{w}</span>' for w in topic_words)
            st.markdown(f"""
            <div class="section-card">
                <div class="topic-header {tc}">◆ {topic_labels[idx]}</div>
                <div class="topic-container">{pills}</div>
            </div>
            """, unsafe_allow_html=True)

# ---- Sentiment Analysis ----
elif page == "Sentiment Analysis":
    st.markdown('<div class="page-hero"><h1>Sentiment Analysis</h1><p>Emotional polarity breakdown of the full corpus.</p></div>', unsafe_allow_html=True)

    sc = df["sentiment"].value_counts()
    clr_map = {"Positive": "#22d3a5", "Neutral": "#facc15", "Negative": "#f87171"}

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card"><h3>Donut Chart</h3>', unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=sc.index, values=sc.values,
            hole=0.58,
            marker=dict(
                colors=[clr_map.get(s, "#6b7280") for s in sc.index],
                line=dict(color="#111827", width=3),
            ),
            textfont=dict(family="DM Sans", size=13, color="#e2e8f0"),
            pull=[0.04] * len(sc),
        ))
        fig_pie.update_layout(**PLOT_LAYOUT, legend=dict(font=dict(color="#94a3b8", size=12)))
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><h3>Bar Chart</h3>', unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            x=sc.index, y=sc.values,
            marker_color=[clr_map.get(s, "#6b7280") for s in sc.index],
            marker_line_width=0,
            text=sc.values,
            textposition="outside",
            textfont=dict(color="#94a3b8", size=12),
        ))
        fig_bar.update_layout(**PLOT_LAYOUT, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---- Impact Level ----
elif page == "Impact Level":
    st.markdown('<div class="page-hero"><h1>Impact Level</h1><p>Derived from sentiment — High, Medium, Low.</p></div>', unsafe_allow_html=True)

    ic = df["impact_level"].value_counts()
    clr_imp = {"High": "#22d3a5", "Medium": "#facc15", "Low": "#f87171"}

    st.markdown('<div class="section-card"><h3>Distribution by Impact</h3>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=ic.index, y=ic.values,
        marker_color=[clr_imp.get(i, "#6b7280") for i in ic.index],
        marker_line_width=0,
        text=ic.values,
        textposition="outside",
        textfont=dict(color="#94a3b8", size=12),
        width=0.5,
    ))
    fig.update_layout(**PLOT_LAYOUT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Source Analysis ----
elif page == "Source Analysis":
    st.markdown('<div class="page-hero"><h1>Source Analysis</h1><p>Which publishers dominate the feed?</p></div>', unsafe_allow_html=True)

    src = df["source_name"].value_counts().head(12)

    st.markdown('<div class="section-card"><h3>Top 12 News Sources</h3>', unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        y=src.index[::-1],
        x=src.values[::-1],
        orientation="h",
        marker=dict(
            color=list(range(len(src))),
            colorscale=[[0, "#0d1f3c"], [0.5, "#4f46e5"], [1, "#a78bfa"]],
            line_width=0,
        ),
        text=src.values[::-1],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig.update_layout(**PLOT_LAYOUT, showlegend=False,
                      margin=dict(l=10, r=60, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Search News ----
elif page == "Search News":
    st.markdown('<div class="page-hero"><h1>Search Articles</h1><p>Filter the corpus by any keyword.</p></div>', unsafe_allow_html=True)

    keyword = st.text_input("", placeholder="🔍  Type a keyword and press Enter…")

    if keyword:
        results = df[df["title"].str.contains(keyword, case=False, na=False)]
        st.markdown(f'<div class="result-badge">⚡ {len(results):,} results for "{keyword}"</div>', unsafe_allow_html=True)
        if not results.empty:
            display_df = results[["title", "sentiment", "impact_level"]].head(25).copy()
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No articles matched that keyword.")
    else:
        st.markdown("""
        <div class="section-card" style="text-align:center; padding: 3rem;">
            <div style="font-size:3rem; margin-bottom:1rem; opacity:0.4;">🔍</div>
            <div style="color:#475569; font-size:0.9rem;">Enter a keyword above to search through all articles</div>
        </div>
        """, unsafe_allow_html=True)