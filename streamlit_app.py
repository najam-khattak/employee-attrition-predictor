import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AttritionIQ — HR Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# DARK PROFESSIONAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800;900&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #080b14 !important;
    color: #e2e8f0 !important;
}
.block-container {
    background: #080b14 !important;
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Navbar / Top Header ── */
.navbar {
    background: linear-gradient(135deg, #080d1f 0%, #0d1a3a 50%, #080d1f 100%);
    border: 1px solid #1e2d55;
    border-radius: 20px;
    padding: 32px 40px 28px 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin-bottom: 32px;
    box-shadow: 0 8px 48px rgba(37,99,235,0.15), 0 2px 8px rgba(0,0,0,0.6);
    position: relative;
    overflow: hidden;
}
.navbar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #7c3aed, #2563eb, #06b6d4, #2563eb, #7c3aed);
    background-size: 200% 100%;
}
.navbar-brand {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
}
.navbar-logo {
    width: 64px; height: 64px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    border-radius: 18px;
    display: flex; align-items: center; justify-content: center;
    font-size: 32px;
    box-shadow: 0 4px 24px rgba(37,99,235,0.4);
    margin-bottom: 6px;
}
.navbar-title {
    font-size: 220px;
    font-weight: 900;
    letter-spacing: 3px;
    word-spacing: 2px;
    margin: 0;
    line-height: 1;
    color: #ffffff !important;
    text-align: center;
    text-transform: uppercase;

    font-family: 'Poppins', sans-serif;

    text-shadow:
        0 0 25px rgba(147, 197, 253, 0.55),
        0 0 50px rgba(99, 102, 241, 0.30),
        0 4px 12px rgba(0,0,0,0.45);
}
.navbar-subtitle {
    font-size: 24px;
    color: #93c5fd !important;
    margin: 18px 0 0 0;
    font-weight: 600;
    letter-spacing: 5px;
    text-align: center;
    text-transform: uppercase;
}
.navbar-badge {
    background: linear-gradient(135deg, #2563eb44, #7c3aed44);
    border: 2px solid #2563eb99;
    border-radius: 28px;
    padding: 14px 36px;
    font-size: 17px;
    font-weight: 700;
    color: #ffffff !important;
    margin-top: 20px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: linear-gradient(135deg, #0d1526 0%, #111827 100%);
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.blue::before  { background: linear-gradient(90deg, #2563eb, #60a5fa); }
.kpi-card.red::before   { background: linear-gradient(90deg, #dc2626, #f87171); }
.kpi-card.green::before { background: linear-gradient(90deg, #16a34a, #4ade80); }
.kpi-card.purple::before{ background: linear-gradient(90deg, #7c3aed, #a78bfa); }

.kpi-icon {
    font-size: 28px;
    margin-bottom: 10px;
    display: block;
}
.kpi-value {
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 12px;
    font-weight: 500;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.kpi-delta {
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
    padding: 3px 8px;
    border-radius: 6px;
    display: inline-block;
}
.delta-bad  { background: #2d0a0a; color: #f87171 !important; }
.delta-good { background: #052e16; color: #4ade80 !important; }

/* ── Section Cards ── */
.card {
    background: linear-gradient(135deg, #0d1526 0%, #111827 100%);
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 24px 26px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid #1e2d45;
}
.card-icon {
    width: 50px; height: 50px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
}
.ci-blue   { background: linear-gradient(135deg, #1d4ed8, #2563eb); border: 2px solid #60a5fa; box-shadow: 0 0 18px #2563eb88; }
.ci-purple { background: linear-gradient(135deg, #6d28d9, #7c3aed); border: 2px solid #a78bfa; box-shadow: 0 0 18px #7c3aed88; }
.ci-green  { background: linear-gradient(135deg, #15803d, #16a34a); border: 2px solid #4ade80; box-shadow: 0 0 18px #16a34a88; }
.ci-red    { background: linear-gradient(135deg, #b91c1c, #dc2626); border: 2px solid #f87171; box-shadow: 0 0 18px #dc262688; }
.ci-yellow { background: linear-gradient(135deg, #b45309, #d97706); border: 2px solid #fbbf24; box-shadow: 0 0 18px #d9770688; }
.ci-teal   { background: linear-gradient(135deg, #0f766e, #0d9488); border: 2px solid #2dd4bf; box-shadow: 0 0 18px #0d948888; }
.ci-pink   { background: linear-gradient(135deg, #9d174d, #be185d); border: 2px solid #f472b6; box-shadow: 0 0 18px #be185d88; }
.ci-orange { background: linear-gradient(135deg, #c2410c, #ea580c); border: 2px solid #fb923c; box-shadow: 0 0 18px #ea580c88; }

.card-title {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    margin: 0 !important;
    letter-spacing: -0.4px;
}
.card-subtitle {
    font-size: 14px;
    color: #64748b !important;
    margin: 5px 0 0 0;
    font-weight: 400;
}

/* ── Input Styling ── */
.stSelectbox label, .stNumberInput label, .stSlider label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #0d1526 !important;
    border: 1px solid #1e2d45 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px #2563eb22 !important;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 28px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 24px rgba(37,99,235,0.45) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(37,99,235,0.6) !important;
}

/* ── Result Banners ── */
.result-high {
    background: linear-gradient(135deg, #2d0a0a, #3b1010);
    border: 1px solid #7f1d1d;
    border-left: 4px solid #ef4444;
    border-radius: 14px;
    padding: 22px 26px;
    margin: 20px 0 16px 0;
}
.result-low {
    background: linear-gradient(135deg, #052e16, #0a3d20);
    border: 1px solid #166534;
    border-left: 4px solid #22c55e;
    border-radius: 14px;
    padding: 22px 26px;
    margin: 20px 0 16px 0;
}
.result-emoji { font-size: 32px; margin-bottom: 8px; display: block; }
.result-title-high { font-size: 22px; font-weight: 800; color: #fca5a5 !important; margin: 0 0 6px; }
.result-title-low  { font-size: 22px; font-weight: 800; color: #86efac !important; margin: 0 0 6px; }
.result-desc { font-size: 14px; color: #94a3b8 !important; margin: 0; line-height: 1.6; }

/* ── Stat Cards (inline) ── */
.stat-card {
    background: #0d1526;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.stat-label {
    font-size: 11px;
    font-weight: 600;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 6px;
}
.stat-value {
    font-size: 24px;
    font-weight: 800;
    line-height: 1;
}

/* ── Factor Pills ── */
.pill-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }
.pill {
    background: #0f1d35;
    border: 1px solid #1e3a6a;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.pill-label { font-size: 10px; font-weight: 700; color: #3b82f6 !important; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 5px; }
.pill-value { font-size: 15px; font-weight: 700; color: #e2e8f0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1526;
    border: 1px solid #1e2d45;
    border-radius: 18px;
    padding: 8px;
    gap: 8px;
    margin-bottom: 28px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 13px !important;
    padding: 18px 42px !important;
    font-weight: 800 !important;
    font-size: 20px !important;
    color: #64748b !important;
    transition: all 0.2s !important;
    letter-spacing: 0.1px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a8a, #2563eb) !important;
    color: white !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important;
    font-size: 20px !important;
}

/* ── Progress ── */
.stProgress > div > div {
    border-radius: 8px !important;
    height: 12px !important;
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
}
.stProgress > div {
    background: #1e2d45 !important;
    border-radius: 8px !important;
    height: 12px !important;
}

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #1e2d45 !important; border-radius: 12px !important; }
iframe { border-radius: 12px !important; }

/* ── Divider ── */
hr { border-color: #1e2d45 !important; margin: 20px 0 !important; }

/* ── Section label ── */
.sec-label {
    font-size: 15px;
    font-weight: 800;
    color: #60a5fa !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 20px;
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a8a44, transparent);
}

/* ── Dashboard insight box ── */
.insight-box {
    background: #0f1d35;
    border: 1px solid #1e3a6a;
    border-radius: 12px;
    padding: 16px 18px;
    margin-top: 12px;
}
.insight-title { font-size: 13px; font-weight: 700; color: #60a5fa !important; margin-bottom: 6px; }
.insight-text  { font-size: 13px; color: #94a3b8 !important; line-height: 1.6; margin: 0; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .navbar { padding: 28px 16px; }
    .navbar-title { font-size: 80px; letter-spacing: 1px; word-spacing: 1px; line-height: 1.1; -webkit-text-fill-color: #ffffff !important; }
    .navbar-subtitle { font-size: 13px; letter-spacing: 2px; }
    .navbar-badge { font-size: 12px; padding: 10px 16px; letter-spacing: 1px; }
    .navbar-logo { width: 52px; height: 52px; font-size: 26px; }
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .kpi-value { font-size: 24px; }
    .stTabs [data-baseweb="tab"] { padding: 10px 12px !important; font-size: 12px !important; }
    .pill-grid { grid-template-columns: 1fr; }
    .card-title { font-size: 17px !important; }
    .sec-label { font-size: 13px; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & LOADERS
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH    = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH   = os.path.join("models", "best_model.pkl")
DROP_COLS    = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
RANDOM_STATE = 42
TEST_SIZE    = 0.2

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_processed_data():
    df = load_raw_data().copy()
    df = df.drop(columns=DROP_COLS)
    le = LabelEncoder()
    df["Attrition"] = le.fit_transform(df["Attrition"])
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test

model  = load_model()
raw_df = load_raw_data()

# Chart style — full white figure + dark readable labels everywhere
def dark_fig(figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("#f8fafc")   # very light grey inside plot area
    ax.tick_params(axis="both",
                   colors="#0f172a",
                   labelcolor="#0f172a",
                   labelsize=10,
                   width=1.2)
    ax.xaxis.label.set_color("#0f172a")
    ax.yaxis.label.set_color("#0f172a")
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    ax.xaxis.label.set_fontweight("600")
    ax.yaxis.label.set_fontweight("600")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")
        spine.set_linewidth(1)
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════════════════════
total       = raw_df.shape[0]
at_risk     = (raw_df["Attrition"] == "Yes").sum()
at_rate     = at_risk / total * 100
avg_income  = raw_df["MonthlyIncome"].mean()
avg_tenure  = raw_df["YearsAtCompany"].mean()

st.markdown(f"""
<div class="navbar">
  <div class="navbar-brand">
    <div class="navbar-logo">🧠</div>
    <p class="navbar-title">AttritionIQ</p>
    <p class="navbar-subtitle">HR Intelligence &amp; Attrition Analytics Platform</p>
    <span class="navbar-badge">⚡ Powered by Logistic Regression &nbsp;|&nbsp; IBM HR Dataset</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Dashboard",
    "🔍  Predict Attrition",
    "📊  Data Insights",
    "🤖  Model Performance",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # KPI Row
    st.markdown("""<div class="sec-label">📌 Key Performance Indicators</div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card blue">
        <span class="kpi-icon">👥</span>
        <div class="kpi-value" style="color:#60a5fa">{total:,}</div>
        <div class="kpi-label">Total Employees</div>
        <span class="kpi-delta delta-good">↑ Full Dataset</span>
      </div>
      <div class="kpi-card red">
        <span class="kpi-icon">⚠️</span>
        <div class="kpi-value" style="color:#f87171">{at_risk}</div>
        <div class="kpi-label">Attrition Cases</div>
        <span class="kpi-delta delta-bad">↑ {at_rate:.1f}% Rate</span>
      </div>
      <div class="kpi-card green">
        <span class="kpi-icon">💵</span>
        <div class="kpi-value" style="color:#4ade80">${avg_income:,.0f}</div>
        <div class="kpi-label">Avg Monthly Income</div>
        <span class="kpi-delta delta-good">↑ Across All Roles</span>
      </div>
      <div class="kpi-card purple">
        <span class="kpi-icon">📅</span>
        <div class="kpi-value" style="color:#a78bfa">{avg_tenure:.1f} yrs</div>
        <div class="kpi-label">Avg Tenure</div>
        <span class="kpi-delta delta-good">↑ Company Average</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: Attrition by Dept + OverTime impact
    st.markdown("""<div class="sec-label">📊 Attrition Overview</div>""", unsafe_allow_html=True)
    r1c1, r1c2, r1c3 = st.columns([1.2, 1.2, 1])

    with r1c1:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-red">🏢</div><div><p class="card-title">Attrition by Department</p><p class="card-subtitle">Which teams lose the most talent</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((5, 3.2))
        dept = raw_df.groupby("Department")["Attrition"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        dept.columns = ["Dept","Rate"]
        dept = dept.sort_values("Rate")
        colors = ["#1e3a8a","#2563eb","#60a5fa"]
        bars = ax.barh(dept["Dept"], dept["Rate"], color=colors, edgecolor="white", linewidth=1.5, height=0.5)
        for bar in bars:
            w = bar.get_width()
            ax.text(w+0.4, bar.get_y()+bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=10, fontweight="700", color="#0f172a")
        ax.set_xlabel("Attrition Rate (%)", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-yellow">⏰</div><div><p class="card-title">OverTime vs Attrition</p><p class="card-subtitle">Impact of overtime on leaving</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((5, 3.2))
        ot = raw_df.groupby(["OverTime","Attrition"]).size().unstack(fill_value=0)
        x = np.arange(len(ot.index))
        w = 0.35
        ax.bar(x-w/2, ot.get("No",0), width=w, color="#22c55e", label="Stayed", edgecolor="white")
        ax.bar(x+w/2, ot.get("Yes",0), width=w, color="#ef4444", label="Left", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(ot.index, color="#0f172a", fontsize=10)
        ax.set_xlabel("OverTime", fontsize=10)
        ax.set_ylabel("Employees", fontsize=10)
        ax.legend(fontsize=9, facecolor="white", labelcolor="#0f172a", edgecolor="#cbd5e1")
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c3:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-purple">🧩</div><div><p class="card-title">Attrition Split</p><p class="card-subtitle">Overall ratio</p></div></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.2), facecolor="white")
        sizes  = [total - at_risk, at_risk]
        colors_pie = ["#22c55e", "#ef4444"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=["Stayed","Left"],
            colors=colors_pie, autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor":"white","linewidth":2},
            textprops={"color":"#0f172a","fontsize":11,"fontweight":"600"}
        )
        for at in autotexts:
            at.set_color("#0f172a"); at.set_fontweight("800"); at.set_fontsize(12)
        ax.set_facecolor("white")
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    # Row 2: Age dist + Income vs Attrition + Job Role

    with r2c1:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-blue">🎂</div><div><p class="card-title">Age Distribution by Attrition</p><p class="card-subtitle">Younger employees leave more</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((6, 3.5))
        for label, color, alpha in [("Yes","#ef4444",0.8), ("No","#22c55e",0.5)]:
            ax.hist(raw_df[raw_df["Attrition"]==label]["Age"], bins=18,
                    alpha=alpha, label=label, color=color, edgecolor="white")
        ax.set_xlabel("Age", fontsize=10); ax.set_ylabel("Count", fontsize=10)
        ax.legend(title="Attrition", fontsize=9, facecolor="white", labelcolor="#0f172a", edgecolor="#cbd5e1", title_fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("""<div class="insight-box"><p class="insight-title">💡 Key Insight</p><p class="insight-text">Employees aged 25–35 show the highest attrition. Early career retention programs are critical.</p></div></div>""", unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-green">💵</div><div><p class="card-title">Income vs Attrition</p><p class="card-subtitle">Lower earners leave more often</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((6, 3.5))
        for label, color, alpha in [("Yes","#ef4444",0.8), ("No","#22c55e",0.5)]:
            ax.hist(raw_df[raw_df["Attrition"]==label]["MonthlyIncome"], bins=20,
                    alpha=alpha, label=label, color=color, edgecolor="white")
        ax.set_xlabel("Monthly Income ($)", fontsize=10); ax.set_ylabel("Count", fontsize=10)
        ax.legend(title="Attrition", fontsize=9, facecolor="white", labelcolor="#0f172a", edgecolor="#cbd5e1", title_fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("""<div class="insight-box"><p class="insight-title">💡 Key Insight</p><p class="insight-text">Employees earning below $5,000/month are 3× more likely to leave. Competitive pay reduces attrition.</p></div></div>""", unsafe_allow_html=True)

    # Row 3: Job Role attrition + Marital status
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-teal">👔</div><div><p class="card-title">Attrition Rate by Job Role</p><p class="card-subtitle">Sales reps at highest risk</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((6, 4))
        role = raw_df.groupby("JobRole")["Attrition"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        role.columns = ["Role","Rate"]
        role = role.sort_values("Rate")
        bar_colors = ["#1e3a8a" if v < 15 else "#2563eb" if v < 25 else "#ef4444" for v in role["Rate"]]
        ax.barh(role["Role"], role["Rate"], color=bar_colors, edgecolor="white", height=0.6)
        ax.set_xlabel("Attrition Rate (%)", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with r3c2:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-purple">💍</div><div><p class="card-title">Attrition by Marital Status</p><p class="card-subtitle">Single employees leave more</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((6, 4))
        ms = raw_df.groupby(["MaritalStatus","Attrition"]).size().unstack(fill_value=0)
        ms_rate = (ms["Yes"] / ms.sum(axis=1) * 100).reset_index()
        ms_rate.columns = ["Status","Rate"]
        ms_rate = ms_rate.sort_values("Rate")
        bar_c = ["#22c55e" if v < 15 else "#f59e0b" if v < 25 else "#ef4444" for v in ms_rate["Rate"]]
        bars = ax.bar(ms_rate["Status"], ms_rate["Rate"], color=bar_c, edgecolor="white", width=0.45)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.5, f"{h:.1f}%",
                    ha="center", fontsize=11, fontweight="700", color="#0f172a")
        ax.set_ylabel("Attrition Rate (%)", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 4: Business Travel + Work Life Balance
    st.markdown("""<div class="sec-label">✈️ Work Conditions</div>""", unsafe_allow_html=True)
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-yellow">✈️</div><div><p class="card-title">Business Travel vs Attrition</p><p class="card-subtitle">Frequent travellers leave more</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((5, 3.2))
        bt = raw_df.groupby("BusinessTravel")["Attrition"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        bt.columns = ["Travel","Rate"]
        bt = bt.sort_values("Rate")
        ax.barh(bt["Travel"], bt["Rate"], color=["#22c55e","#f59e0b","#ef4444"], edgecolor="white", height=0.45)
        ax.set_xlabel("Attrition Rate (%)", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with r4c2:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-green">⚖️</div><div><p class="card-title">Work-Life Balance vs Attrition</p><p class="card-subtitle">Poor balance drives departures</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((5, 3.2))
        wlb = raw_df.groupby("WorkLifeBalance")["Attrition"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        wlb.columns = ["WLB","Rate"]
        bar_c = ["#ef4444","#f59e0b","#22c55e","#22c55e"]
        bars = ax.bar(wlb["WLB"].astype(str), wlb["Rate"], color=bar_c[:len(wlb)], edgecolor="white", width=0.45)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f"{h:.1f}%",
                    ha="center", fontsize=10, fontweight="700", color="#0f172a")
        ax.set_xlabel("Work-Life Balance (1=Low, 4=High)", fontsize=10)
        ax.set_ylabel("Attrition Rate (%)", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT (all 30 features restored)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""<div class="sec-label">🔍 Employee Risk Assessment</div>""", unsafe_allow_html=True)

    # ── Personal Info ─────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-blue">👤</div><div><p class="card-title">Personal Information</p><p class="card-subtitle">Basic demographic details of the employee</p></div></div>', unsafe_allow_html=True)
    p1,p2,p3,p4 = st.columns(4)
    age                = p1.number_input("🎂 Age", 18, 60, 35)
    gender             = p2.selectbox("⚧ Gender", ["Female","Male"])
    marital_status     = p3.selectbox("💍 Marital Status", ["Divorced","Married","Single"])
    distance_from_home = p4.number_input("🏠 Distance From Home (km)", 1, 30, 5)
    p5,p6 = st.columns(2)
    education          = p5.selectbox("🎓 Education (1=Below College … 5=Doctor)", [1,2,3,4,5])
    education_field    = p6.selectbox("📚 Education Field", ["Human Resources","Life Sciences","Marketing","Medical","Other","Technical Degree"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Job Info ──────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-purple">💼</div><div><p class="card-title">Job Information</p><p class="card-subtitle">Role, department and work conditions</p></div></div>', unsafe_allow_html=True)
    j1,j2,j3 = st.columns(3)
    department       = j1.selectbox("🏢 Department", ["Human Resources","Research & Development","Sales"])
    job_role         = j2.selectbox("👔 Job Role", ["Healthcare Representative","Human Resources","Laboratory Technician","Manager","Manufacturing Director","Research Director","Research Scientist","Sales Executive","Sales Representative"])
    job_level        = j3.selectbox("📶 Job Level (1–5)", [1,2,3,4,5])
    j4,j5,j6 = st.columns(3)
    job_satisfaction = j4.selectbox("😊 Job Satisfaction (1–4)", [1,2,3,4])
    job_involvement  = j5.selectbox("🎯 Job Involvement (1–4)", [1,2,3,4])
    business_travel  = j6.selectbox("✈️ Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
    j7, = st.columns(1)
    overtime         = j7.selectbox("⏰ OverTime", ["No","Yes"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Compensation & Growth ─────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-green">💰</div><div><p class="card-title">Compensation & Growth</p><p class="card-subtitle">Salary, incentives and career progression</p></div></div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    monthly_income        = c1.number_input("💵 Monthly Income ($)", 1000, 20000, 5000)
    daily_rate            = c2.number_input("📆 Daily Rate", 100, 1500, 800)
    hourly_rate           = c3.number_input("⏱️ Hourly Rate", 30, 100, 65)
    c4,c5,c6 = st.columns(3)
    monthly_rate          = c4.number_input("📅 Monthly Rate", 2000, 27000, 14000)
    percent_salary_hike   = c5.number_input("📊 Salary Hike (%)", 11, 25, 14)
    stock_option          = c6.selectbox("📈 Stock Option Level (0–3)", [0,1,2,3])
    c7, = st.columns(1)
    years_since_promotion = c7.number_input("🚀 Years Since Last Promotion", 0, 15, 1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Satisfaction & Ratings ────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-yellow">⭐</div><div><p class="card-title">Satisfaction & Ratings</p><p class="card-subtitle">Workplace satisfaction scores — 1 Low → 4 High</p></div></div>', unsafe_allow_html=True)
    s1,s2,s3,s4 = st.columns(4)
    environment_sat   = s1.selectbox("🌿 Environment Satisfaction (1–4)", [1,2,3,4])
    work_life_balance = s2.selectbox("⚖️ Work Life Balance (1–4)", [1,2,3,4])
    relationship_sat  = s3.selectbox("🤝 Relationship Satisfaction (1–4)", [1,2,3,4])
    performance_rating = s4.selectbox("🏅 Performance Rating (1–4)", [1,2,3,4])
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Experience & Tenure ───────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-teal">📅</div><div><p class="card-title">Experience & Tenure</p><p class="card-subtitle">Work history and time at the company</p></div></div>', unsafe_allow_html=True)
    e1,e2,e3 = st.columns(3)
    total_working_years   = e1.number_input("🗂️ Total Working Years", 0, 40, 10)
    years_at_company      = e2.number_input("🏛️ Years At Company", 0, 40, 5)
    years_in_role         = e3.number_input("🪑 Years In Current Role", 0, 18, 3)
    e4,e5,e6 = st.columns(3)
    years_with_manager    = e4.number_input("👨‍💼 Years With Current Manager", 0, 17, 3)
    num_companies         = e5.number_input("🏭 Companies Worked", 0, 10, 2)
    training_times        = e6.number_input("📖 Training Times Last Year", 0, 6, 3)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Run Attrition Risk Assessment", use_container_width=True)

    if predict_clicked:
        raw_input = {
            "Age": age, "BusinessTravel": business_travel,
            "DailyRate": daily_rate, "Department": department,
            "DistanceFromHome": distance_from_home, "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": environment_sat, "Gender": gender,
            "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
            "JobLevel": job_level, "JobRole": job_role,
            "JobSatisfaction": job_satisfaction, "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies, "OverTime": overtime,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_sat,
            "StockOptionLevel": stock_option,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": years_with_manager,
        }
        input_df = pd.DataFrame([raw_input])
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-high">
              <span class="result-emoji">⚠️</span>
              <p class="result-title-high">High Attrition Risk Detected</p>
              <p class="result-desc">This employee has a <strong style="color:#fca5a5">{probability:.0%} probability</strong>
              of leaving the organization. Immediate HR intervention is recommended —
              consider reviewing compensation, workload, and career growth opportunities.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-low">
              <span class="result-emoji">✅</span>
              <p class="result-title-low">Low Attrition Risk</p>
              <p class="result-desc">This employee has only a <strong style="color:#86efac">{probability:.0%} probability</strong>
              of leaving. They appear stable and engaged. Continue supporting their growth and recognition.</p>
            </div>""", unsafe_allow_html=True)

        # Stat cards
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            color = "#f87171" if prediction==1 else "#4ade80"
            label = "Will Leave ⚠️" if prediction==1 else "Will Stay ✅"
            st.markdown(f'<div class="stat-card"><div class="stat-label">Prediction</div><div class="stat-value" style="color:{color}">{label}</div></div>', unsafe_allow_html=True)
        with sc2:
            color = "#f87171" if probability > 0.5 else "#4ade80"
            st.markdown(f'<div class="stat-card"><div class="stat-label">Risk Probability</div><div class="stat-value" style="color:{color}">{probability:.1%}</div></div>', unsafe_allow_html=True)
        with sc3:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Model</div><div class="stat-value" style="color:#a78bfa; font-size:15px;">Logistic Regression</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**🎯 Attrition Risk Meter — {probability:.1%}**")
        st.progress(float(probability))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">💡 Top Influencing Factors</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="pill-grid">
          <div class="pill"><div class="pill-label">⏰ OverTime</div><div class="pill-value">{overtime}</div></div>
          <div class="pill"><div class="pill-label">💍 Marital Status</div><div class="pill-value">{marital_status}</div></div>
          <div class="pill"><div class="pill-label">💵 Monthly Income</div><div class="pill-value">${monthly_income:,}</div></div>
          <div class="pill"><div class="pill-label">🎂 Age</div><div class="pill-value">{age} yrs</div></div>
          <div class="pill"><div class="pill-label">📈 Stock Options</div><div class="pill-value">Level {stock_option}</div></div>
          <div class="pill"><div class="pill-label">✈️ Business Travel</div><div class="pill-value">{business_travel}</div></div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("""<div class="sec-label">📊 Dataset Deep Dive</div>""", unsafe_allow_html=True)

    i1, i2, i3, i4 = st.columns(4)
    stats = [
        ("👥 Total Records",   f"{raw_df.shape[0]:,}",      "#60a5fa"),
        ("📋 Total Features",  f"{raw_df.shape[1]}",         "#a78bfa"),
        ("⚠️ Attrition Cases", f"{at_risk}",                 "#f87171"),
        ("📉 Attrition Rate",  f"{at_rate:.1f}%",            "#f87171"),
    ]
    for col, (label, val, color) in zip([i1,i2,i3,i4], stats):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-blue">📌</div><div><p class="card-title">Attrition Count</p><p class="card-subtitle">Yes vs No distribution</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((5, 3.5))
        counts = raw_df["Attrition"].value_counts()
        bars = ax.bar(counts.index, counts.values, color=["#22c55e","#ef4444"], edgecolor="white", width=0.45)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+8, str(v), ha="center", fontsize=12, fontweight="800", color="#0f172a")
        ax.set_ylabel("Count", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with ic2:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-purple">🎓</div><div><p class="card-title">Education Field vs Attrition</p><p class="card-subtitle">Which backgrounds leave more</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((5, 3.5))
        ef = raw_df.groupby("EducationField")["Attrition"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        ef.columns = ["Field","Rate"]
        ef = ef.sort_values("Rate")
        ax.barh(ef["Field"], ef["Rate"], color="#7c3aed", edgecolor="white", height=0.5)
        ax.set_xlabel("Attrition Rate (%)", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    ic3, ic4 = st.columns(2)
    with ic3:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-green">😊</div><div><p class="card-title">Job Satisfaction vs Attrition</p><p class="card-subtitle">Lower satisfaction = higher risk</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((5, 3.5))
        js = raw_df.groupby("JobSatisfaction")["Attrition"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        js.columns = ["Sat","Rate"]
        bar_c = ["#ef4444","#f59e0b","#22c55e","#22c55e"]
        bars = ax.bar(js["Sat"].astype(str), js["Rate"], color=bar_c[:len(js)], edgecolor="white", width=0.45)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f"{h:.1f}%", ha="center", fontsize=10, fontweight="700", color="#0f172a")
        ax.set_xlabel("Job Satisfaction Level", fontsize=10)
        ax.set_ylabel("Attrition Rate (%)", fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with ic4:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-red">🏛️</div><div><p class="card-title">Years At Company vs Attrition</p><p class="card-subtitle">Tenure distribution of leavers</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((5, 3.5))
        for label, color, alpha in [("Yes","#ef4444",0.85), ("No","#22c55e",0.5)]:
            ax.hist(raw_df[raw_df["Attrition"]==label]["YearsAtCompany"],
                    bins=15, alpha=alpha, label=label, color=color, edgecolor="white")
        ax.set_xlabel("Years At Company", fontsize=10); ax.set_ylabel("Count", fontsize=10)
        ax.legend(title="Attrition", fontsize=9, facecolor="white", labelcolor="#0f172a", edgecolor="#cbd5e1", title_fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-teal">🔥</div><div><p class="card-title">Feature Correlation Heatmap</p><p class="card-subtitle">Relationships between all numeric features</p></div></div>', unsafe_allow_html=True)
    numeric_df = raw_df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in DROP_COLS if c in raw_df.columns], errors="ignore")
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
    ax.set_facecolor("white")
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
    sns.heatmap(numeric_df.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                annot_kws={"size":7, "color":"#1e293b"},
                ax=ax, cbar_kws={"shrink":0.8})
    ax.tick_params(colors="#1e293b", labelcolor="#0f172a", labelsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.markdown("""<div class="sec-label">🤖 Model Evaluation Report</div>""", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-purple">🤖</div><div><p class="card-title">Logistic Regression — Test Set Results</p><p class="card-subtitle">Evaluated on 20% hold-out test set (294 employees)</p></div></div>', unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = get_processed_data()
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, output_dict=True)

    pm1, pm2, pm3, pm4 = st.columns(4)
    perf_metrics = [
        ("🎯 Accuracy",            f"{report['accuracy']:.4f}",                     "#60a5fa"),
        ("📐 Macro F1 Score",      f"{report['macro avg']['f1-score']:.4f}",         "#a78bfa"),
        ("🔍 Attrition Recall",    f"{report['1']['recall']:.4f}",                  "#f87171"),
        ("✅ Attrition Precision",  f"{report['1']['precision']:.4f}",               "#4ade80"),
    ]
    for col, (label, val, color) in zip([pm1,pm2,pm3,pm4], perf_metrics):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    mp1, mp2 = st.columns(2)

    with mp1:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-blue">🟦</div><div><p class="card-title">Confusion Matrix</p><p class="card-subtitle">Actual vs Predicted classifications</p></div></div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="white")
        ax.set_facecolor("white")
        cm = confusion_matrix(y_test, y_pred)
        cmap = sns.light_palette("#2563eb", as_cmap=True)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=["No Attrition","Attrition"],
                    yticklabels=["No Attrition","Attrition"],
                    linewidths=3, linecolor="white",
                    annot_kws={"size":16,"weight":"bold","color":"#1e293b"}, ax=ax)
        ax.set_xlabel("Predicted", fontsize=12, color="#0f172a", fontweight="600")
        ax.set_ylabel("Actual", fontsize=12, color="#0f172a", fontweight="600")
        ax.tick_params(colors="#1e293b", labelcolor="#0f172a", labelsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with mp2:
        st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-green">📈</div><div><p class="card-title">ROC Curve</p><p class="card-subtitle">Model discriminating power</p></div></div>', unsafe_allow_html=True)
        fig, ax = dark_fig((5, 4))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0,1],[0,1], color="#94a3b8", lw=1.5, linestyle="--", label="Random")
        ax.fill_between(fpr, tpr, alpha=0.12, color="#2563eb")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", fontsize=10, facecolor="white", labelcolor="#0f172a", edgecolor="#cbd5e1")
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-yellow">🏆</div><div><p class="card-title">Top 15 Most Important Features</p><p class="card-subtitle">Logistic Regression coefficient magnitudes</p></div></div>', unsafe_allow_html=True)
    coef          = model.named_steps["model"].coef_[0]
    feature_names = model.feature_names_in_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(coef)})
    importance_df = importance_df.sort_values("Importance", ascending=True).tail(15)
    coef_vals = coef[[list(feature_names).index(f) for f in importance_df["Feature"]]]
    bar_colors = ["#ef4444" if c > 0 else "#2563eb" for c in coef_vals]
    fig, ax = dark_fig((10, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"],
            color=bar_colors, edgecolor="white", linewidth=1.5, height=0.6)
    ax.set_xlabel("Absolute Coefficient Value", fontsize=10)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("""
    <p style="font-size:12px; color:#475569; margin-top:6px;">
    🔴 Red = increases attrition risk &nbsp;|&nbsp; 🔵 Blue = decreases attrition risk
    </p></div>""", unsafe_allow_html=True)

    # Full report table
    st.markdown('<div class="card"><div class="card-header"><div class="card-icon ci-teal">📋</div><div><p class="card-title">Full Classification Report</p><p class="card-subtitle">Precision, Recall, F1 for each class</p></div></div>', unsafe_allow_html=True)
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center; padding:20px; margin-top:24px; border-top:1px solid #1e2d45;">
        <p style="font-size:13px; color:#334155; margin:0;">
            🧠 <strong style="color:#475569">AttritionIQ</strong> &nbsp;|&nbsp;
            Built with Streamlit &amp; Scikit-learn &nbsp;|&nbsp;
            Logistic Regression Model &nbsp;|&nbsp;
            IBM HR Analytics Dataset &nbsp;|&nbsp;
            © 2025
        </p>
    </div>
    """, unsafe_allow_html=True)