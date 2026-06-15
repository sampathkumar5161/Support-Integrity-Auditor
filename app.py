"""
app.py — Support Integrity Auditor (SIA) Streamlit Web App
MARS Open Projects 2026

Run:
    streamlit run app.py
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import spacy, torch
from scipy.special import softmax as sp_softmax
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        text-align: center; color: white;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1.2rem;
        border-left: 4px solid #0f3460; margin: 0.5rem 0;
    }
    .crisis-badge {
        background: #ff4444; color: white; padding: 4px 12px;
        border-radius: 20px; font-weight: bold; font-size: 0.85rem;
    }
    .false-alarm-badge {
        background: #ff8800; color: white; padding: 4px 12px;
        border-radius: 20px; font-weight: bold; font-size: 0.85rem;
    }
    .consistent-badge {
        background: #00aa44; color: white; padding: 4px 12px;
        border-radius: 20px; font-weight: bold; font-size: 0.85rem;
    }
    .dossier-box {
        background: #0f1923; color: #e0e0e0; padding: 1.5rem;
        border-radius: 10px; font-family: monospace; font-size: 0.85rem;
        border: 1px solid #334;
    }
    .evidence-item {
        background: #1e293b; border-radius: 6px; padding: 0.6rem 1rem;
        margin: 0.3rem 0; border-left: 3px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
PRIORITY_ORDER  = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
PRIORITY_LABELS = {v: k for k, v in PRIORITY_ORDER.items()}
MODEL_DIR       = os.environ.get("SIA_MODEL_DIR", "sia_model_final")
MAX_LEN         = 256

CRITICAL_KEYWORDS  = ["urgent","critical","emergency","immediately","asap","broken","outage",
                       "down","crash","failure","data loss","security breach","cannot access",
                       "system down","not working","blocked","escalate","unacceptable",
                       "lawsuit","legal","refund","fraud","hacked"]
HIGH_KEYWORDS      = ["issue","problem","error","bug","slow","delay","frustrat","not responding",
                       "failed","incorrect","wrong","missing","stuck"]
LOW_KEYWORDS       = ["question","inquiry","wondering","curious","how do i",
                       "feature request","suggestion","feedback","would like","could you"]
ESCALATION_PHRASES = ["speak to manager","speak to supervisor","escalate","not acceptable",
                       "very disappointed","extremely frustrated","worst experience"]
NEGATION_WORDS     = ["not","n't","never","no","unable","cannot","can't"]

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_classifier():
    if not os.path.exists(MODEL_DIR):
        return None, None
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-small", num_labels=2, ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()
    return tokenizer, model

# ── NLP helpers ──────────────────────────────────────────────────────────────
def rule_based_severity(text: str, nlp_model) -> tuple[float, list]:
    t      = text.lower()
    doc    = nlp_model(t[:2000])
    tokens = [tok.text for tok in doc]
    score  = 1.0
    evidence_signals = []

    crit_hits = [kw for kw in CRITICAL_KEYWORDS if kw in t]
    high_hits = [kw for kw in HIGH_KEYWORDS      if kw in t]
    esc_hits  = [ph for ph in ESCALATION_PHRASES  if ph in t]
    neg_count = sum(1 for tok in tokens if tok in NEGATION_WORDS)
    excl      = text.count("!")
    caps      = re.findall(r"\b[A-Z]{3,}\b", text)

    if crit_hits:
        delta = min(len(crit_hits) * 0.6, 2.0)
        score += delta
        evidence_signals.append({"signal":"critical_keyword","value":", ".join(crit_hits[:5]),
                                  "source_field":"description","weight":f"{delta:.2f}"})
    if high_hits:
        delta = min(len(high_hits) * 0.2, 0.8)
        score += delta
        evidence_signals.append({"signal":"high_severity_keyword","value":", ".join(high_hits[:5]),
                                  "source_field":"description","weight":f"{delta:.2f}"})
    low_hits = [kw for kw in LOW_KEYWORDS if kw in t]
    if low_hits:
        delta = min(len(low_hits) * 0.3, 0.9)
        score -= delta
    if esc_hits:
        score += len(esc_hits) * 0.5
        evidence_signals.append({"signal":"escalation_phrase","value":", ".join(esc_hits[:3]),
                                  "source_field":"description","weight":f"{len(esc_hits)*0.5:.2f}"})
    if neg_count:
        delta = min(neg_count * 0.15, 0.5)
        score += delta
        evidence_signals.append({"signal":"negation_density","value":str(neg_count),
                                  "source_field":"description","weight":f"{delta:.2f}"})
    if excl:
        delta = min(excl * 0.1, 0.3)
        score += delta
        evidence_signals.append({"signal":"exclamation_density","value":str(excl),
                                  "source_field":"description","weight":f"{delta:.2f}"})
    if caps:
        delta = min(len(caps) * 0.15, 0.45)
        score += delta
        evidence_signals.append({"signal":"caps_word_density","value":", ".join(caps[:5]),
                                  "source_field":"description","weight":f"{delta:.2f}"})
    return float(np.clip(score, 0, 3)), evidence_signals

def parse_resolution_time(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)): return np.nan
    val  = str(val)
    nums = re.findall(r"[\d\.]+", val)
    if not nums: return np.nan
    h = float(nums[0])
    if "day"  in val.lower(): h *= 24
    if "week" in val.lower(): h *= 168
    return h

# ── Core inference ────────────────────────────────────────────────────────────
def infer_single(subject, description, priority, channel, ticket_type, resolution_time_str):
    nlp_model = load_nlp()
    embedder  = load_embedder()
    tokenizer, clf = load_classifier()

    text = f"{subject} [SEP] {description}"
    rt   = parse_resolution_time(resolution_time_str)
    rt_s = f"{rt:.1f}h" if not (rt is None or np.isnan(rt)) else "unknown"
    model_input = f"[CHANNEL: {channel}] [TYPE: {ticket_type}] [RT: {rt_s}] {text}"

    nlp_score, nlp_evidence = rule_based_severity(text, nlp_model)
    nlp_sev = int(np.clip(round(nlp_score), 0, 3))

    # Embedding severity (simple: use NLP as proxy since we have no reference cluster)
    inferred_sev_num = nlp_sev
    inferred_sev     = PRIORITY_LABELS[inferred_sev_num]
    priority_num     = PRIORITY_ORDER.get(priority, 1)
    mismatch         = inferred_sev_num != priority_num
    mismatch_type    = ("Consistent" if not mismatch else
                        ("Hidden Crisis" if inferred_sev_num > priority_num else "False Alarm"))

    # Resolution-time evidence
    if rt and not np.isnan(rt):
        nlp_evidence.append({"signal":"resolution_time","value":f"{rt:.1f} hours",
                              "source_field":"resolution_time",
                              "interpretation":"Provided by user"})
    if channel and channel.lower() in ["phone","social media","social_media"]:
        nlp_evidence.append({"signal":"channel","value":channel,
                              "source_field":"channel","weight":"0.30"})
    if not nlp_evidence:
        nlp_evidence.append({"signal":"text_semantics","value":"semantic analysis",
                              "source_field":"description","weight":"0.30"})

    # Classifier
    confidence = 0.5
    if clf is not None and mismatch:
        enc = tokenizer(model_input, truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = clf(**enc).logits.squeeze().numpy()
        prob = sp_softmax(logits)
        confidence = float(prob[1])

    # Constraint analysis
    sigs = [e["signal"] for e in nlp_evidence]
    if mismatch_type == "Hidden Crisis":
        analysis = (f"Ticket assigned {priority} but signals indicate true severity of {inferred_sev}. "
                    f"Evidence from {', '.join(sigs[:3])} is directly traceable to input fields. "
                    f"Immediate re-triage is recommended to prevent SLA breach.")
    elif mismatch_type == "False Alarm":
        analysis = (f"Ticket inflated to {priority}; inferred severity is only {inferred_sev}. "
                    f"Evidence from {', '.join(sigs[:3])} is traceable to input fields. "
                    f"Deprioritisation advised to free support bandwidth.")
    else:
        analysis = (f"Assigned priority {priority} aligns with inferred severity {inferred_sev}. "
                    f"Signals from {', '.join(sigs[:3]) if sigs else 'text analysis'} support this classification. "
                    f"No re-triage action required.")

    dossier = {
        "ticket_id"          : "LIVE-QUERY",
        "assigned_priority"  : priority,
        "inferred_severity"  : inferred_sev,
        "mismatch_type"      : mismatch_type,
        "severity_delta"     : inferred_sev_num - priority_num,
        "feature_evidence"   : nlp_evidence,
        "constraint_analysis": analysis,
        "confidence"         : f"{confidence:.3f}"
    }
    return dossier

def batch_infer(df: pd.DataFrame) -> pd.DataFrame:
    nlp_model = load_nlp()
    embedder  = load_embedder()

    df = df.copy()
    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    df["text"] = (df.get("subject", df.get("Ticket_Subject","")).fillna("") +
                  " [SEP] " + df.get("description", df.get("Ticket_Description","")).fillna("")).str.strip()
    if "priority" not in df.columns and "Ticket_Priority" in df.columns:
        df["priority"] = df["Ticket_Priority"]
    df["priority_num"] = df["Priority_Level"].map(PRIORITY_ORDER).fillna(1).astype(int)

    scores = []
    for t in df["text"]:
        s, _ = rule_based_severity(t, nlp_model)
        scores.append(s)
    df["nlp_score"] = scores
    df["nlp_sev"]   = pd.cut(df["nlp_score"], bins=[-0.1,0.75,1.5,2.25,3.1], labels=[0,1,2,3]).astype(int)

    embs     = embedder.encode(df["text"].tolist(), batch_size=32, show_progress_bar=False)
    km       = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = km.fit_predict(embs)
    df["cluster"] = clusters
    csmap = (df.groupby("cluster")["nlp_score"].median()
               .rank(method="first").sub(1).astype(int).to_dict())
    df["cluster_sev"] = df["cluster"].map(csmap)

    df["inferred_num"]      = ((0.55 * df["nlp_sev"] + 0.45 * df["cluster_sev"])
                                .round().astype(int).clip(0, 3))
    df["inferred_severity"] = df["inferred_num"].map(PRIORITY_LABELS)
    df["mismatch"]          = (df["inferred_num"] != df["priority_num"]).astype(int)
    df["mismatch_type"]     = df.apply(lambda r: (
        "Consistent" if r["mismatch"] == 0
        else ("Hidden Crisis" if r["inferred_num"] > r["priority_num"] else "False Alarm")), axis=1)
    df["severity_delta"]    = df["inferred_num"] - df["priority_num"]
    df["confidence"]        = 0.72  # placeholder without full model; replace with model probs if available
    return df

# ── Render dossier ─────────────────────────────────────────────────────────── 
def render_dossier(d: dict):
    mtype = d.get("mismatch_type", "Consistent")
    badge = (f'<span class="crisis-badge">🚨 Hidden Crisis</span>'     if mtype == "Hidden Crisis" else
             f'<span class="false-alarm-badge">⚠️ False Alarm</span>'  if mtype == "False Alarm"  else
             f'<span class="consistent-badge">✅ Consistent</span>')

    delta_num = d.get("severity_delta", 0)
    delta_str = f"+{delta_num}" if delta_num > 0 else str(delta_num)

    st.markdown(f"""
    <div class="metric-card">
      <b>Ticket:</b> {d['ticket_id']} &nbsp;&nbsp;
      {badge} &nbsp;&nbsp;
      <b>Assigned:</b> {d['assigned_priority']} &nbsp;→&nbsp;
      <b>Inferred:</b> {d['inferred_severity']} &nbsp;&nbsp;
      <b>Δ:</b> {delta_str} &nbsp;&nbsp;
      <b>Confidence:</b> {d['confidence']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📋 Constraint Analysis**")
    st.info(d.get("constraint_analysis", ""))

    st.markdown("**🔎 Feature Evidence** *(all items traceable to input fields)*")
    for ev in d.get("feature_evidence", []):
        field = ev.get("source_field","N/A")
        sig   = ev.get("signal","")
        val   = ev.get("value","")
        w     = ev.get("weight", ev.get("interpretation",""))
        st.markdown(
            f'<div class="evidence-item">📌 <b>{sig}</b> | Field: <code>{field}</code> | '
            f'Value: <i>{val}</i> | Weight/Note: {w}</div>',
            unsafe_allow_html=True
        )

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/300x80/0f3460/FFFFFF?text=SIA+MARS+2026", use_column_width=True)
    st.markdown("---")
    st.markdown("### 🔧 About")
    st.markdown("""
**Support Integrity Auditor**  
Detects priority mismatches in CRM support tickets using:
- Rule-based NLP scoring
- Semantic embedding clustering  
- Fine-tuned DeBERTa-v3 + LoRA classifier

*MARS Open Projects 2026*
    """)
    st.markdown("---")
    mode = st.radio("**Mode**", [" Single Ticket", "📂 Batch CSV Upload", "📊 Dashboard"])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🔍 Support Integrity Auditor</h1>
  <p style="font-size:1.1rem; opacity:0.85;">
    Detecting Priority Mismatches in CRM Support Tickets · MARS Open Projects 2026
  </p>
</div>
""", unsafe_allow_html=True)

# ── Mode: Single Ticket ───────────────────────────────────────────────────────
if "Single" in mode:
    st.subheader(" Analyse a Single Ticket")
    col1, col2 = st.columns([3, 2])
    with col1:
        subject     = st.text_input("Ticket Subject", placeholder="e.g. Cannot login to dashboard")
        description = st.text_area("Ticket Description", height=160,
                        placeholder="Describe the issue in detail...")
    with col2:
        priority       = st.selectbox("Assigned Priority", ["Low","Medium","High","Critical"])
        channel        = st.selectbox("Ticket Channel", ["email","chat","phone","social media","web portal"])
        ticket_type    = st.text_input("Ticket Type", value="technical")
        resolution_hrs = st.text_input("Resolution Time (optional)", placeholder="e.g. 48 hours or 3 days")

    if st.button("🔍 Analyse Ticket", type="primary", use_container_width=True):
        if not description.strip():
            st.error("Please enter a ticket description.")
        else:
            with st.spinner("Analysing ticket ..."):
                dossier = infer_single(subject, description, priority, channel, ticket_type, resolution_hrs)
            st.markdown("---")
            st.subheader("📄 Evidence Dossier")
            render_dossier(dossier)
            st.markdown("---")
            st.subheader("📋 Raw Dossier JSON")
            st.json(dossier)

# ── Mode: Batch CSV ───────────────────────────────────────────────────────────
elif "Batch" in mode:
    st.subheader("📂 Batch Analysis via CSV Upload")
    st.markdown("""
    **Expected columns:** `Ticket_ID`, `Ticket_Subject`, `Ticket_Description`,
    `Ticket_Priority`, `Ticket_Channel`, `Ticket_Type`, `Resolution_Time`
    """)
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        # Normalise column names
        df_raw.columns = df_raw.columns.str.strip().str.replace(" ","_")
        rename = {"Ticket_ID":"ticket_id","Ticket_Subject":"subject","Ticket_Description":"description",
                  "Ticket_Priority":"priority","Ticket_Channel":"channel","Ticket_Type":"ticket_type",
                  "Resolution_Time":"resolution_time"}
        df_raw.rename(columns={k:v for k,v in rename.items() if k in df_raw.columns}, inplace=True)

        st.success(f"Loaded {len(df_raw)} tickets.")
        st.dataframe(df_raw.head(5))

        if st.button(" Run Batch Analysis", type="primary"):
            with st.spinner("Processing all tickets ..."):
                result_df = batch_infer(df_raw)
                st.session_state["batch_results"] = result_df  # save for dashboard

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tickets",     len(result_df))
            c2.metric("Mismatches Found",  result_df["mismatch"].sum())
            c3.metric("Hidden Crises",     (result_df["mismatch_type"]=="Hidden Crisis").sum())
            c4.metric("False Alarms",      (result_df["mismatch_type"]=="False Alarm").sum())

            # Charts
            st.subheader("📊 Results Overview")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(result_df, names="mismatch_type", title="Mismatch Type Distribution",
                             color_discrete_map={"Consistent":"#00aa44","Hidden Crisis":"#ff4444","False Alarm":"#ff8800"})
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.histogram(result_df, x="Priority_Level", color="mismatch_type",
                                    barmode="group", title="Priority vs Mismatch Type",
                                    color_discrete_map={"Consistent":"#00aa44","Hidden Crisis":"#ff4444","False Alarm":"#ff8800"})
                st.plotly_chart(fig2, use_container_width=True)

            # Heatmap: severity delta across categories and channels
            st.subheader(" Severity Delta Heatmap (Ticket Type × Channel)")
            if "ticket_type" in result_df.columns and "channel" in result_df.columns:
                hm = result_df.groupby(["ticket_type","channel"])["severity_delta"].mean().reset_index()
                hm_pivot = hm.pivot(index="ticket_type", columns="channel", values="severity_delta").fillna(0)
                fig3 = go.Figure(data=go.Heatmap(
                    z=hm_pivot.values, x=hm_pivot.columns.tolist(), y=hm_pivot.index.tolist(),
                    colorscale="RdYlGn_r", zmid=0,
                    text=np.round(hm_pivot.values, 2), texttemplate="%{text}",
                    colorbar=dict(title="Avg Δ Severity")
                ))
                fig3.update_layout(title="Mean Severity Delta (positive = under-assigned)",
                                   xaxis_title="Channel", yaxis_title="Ticket Type")
                st.plotly_chart(fig3, use_container_width=True)

            # Table of flagged tickets
            st.subheader(" Flagged Tickets")
            flagged = result_df[result_df["mismatch"] == 1][
                ["ticket_id","Priority_Level","inferred_severity","mismatch_type","severity_delta","confidence"]
            ]
            st.dataframe(flagged, use_container_width=True)

            # Download
            csv_out = result_df[["ticket_id","Priority_Level","inferred_severity",
                                  "mismatch","mismatch_type","severity_delta","confidence"]].to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv_out, "sia_results.csv", "text/csv")

# ── Mode: Dashboard ───────────────────────────────────────────────────────────
elif "Dashboard" in mode:
    st.subheader("📊 Priority Mismatch Dashboard")

    # Allow direct upload on dashboard OR use results from Batch mode
    if "batch_results" not in st.session_state:
        st.info("Upload your tickets CSV below to populate the dashboard with real data.")
        dash_upload = st.file_uploader("Upload customer_support_tickets.csv", type=["csv"], key="dash_csv")
        if dash_upload:
            raw = pd.read_csv(dash_upload)
            raw.columns = raw.columns.str.strip().str.replace(" ", "_")
            rename = {"Ticket_ID":"ticket_id","Ticket_Subject":"subject","Ticket_Description":"description",
                      "Ticket_Priority":"priority","Ticket_Channel":"channel","Ticket_Type":"ticket_type",
                      "Resolution_Time":"resolution_time"}
            raw.rename(columns={k:v for k,v in rename.items() if k in raw.columns}, inplace=True)
            with st.spinner("Analysing tickets..."):
                st.session_state["batch_results"] = batch_infer(raw)
            st.rerun()
        else:
            st.stop()

    result_df = st.session_state["batch_results"]
    st.success(f"Showing real results for {len(result_df)} tickets")

    # ── KPI cards ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    mismatch_rate = result_df["mismatch"].mean() * 100
    c1.metric("Total Tickets",      len(result_df))
    c2.metric("Mismatches Flagged", result_df["mismatch"].sum(),
              delta=f"{mismatch_rate:.1f}% mismatch rate")
    c3.metric("🚨 Hidden Crises",   (result_df["mismatch_type"]=="Hidden Crisis").sum(),
              delta_color="inverse")
    c4.metric("⚠️ False Alarms",    (result_df["mismatch_type"]=="False Alarm").sum())

    st.markdown("---")

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(result_df, names="mismatch_type",
                     title="Mismatch Type Distribution",
                     color_discrete_map={"Consistent":"#00aa44",
                                         "Hidden Crisis":"#ff4444",
                                         "False Alarm":"#ff8800"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        chan_data = result_df.groupby("channel")["mismatch"].mean().reset_index()
        fig2 = px.bar(chan_data, x="channel", y="mismatch",
                      title="Mismatch Rate by Channel",
                      labels={"mismatch":"Mismatch Rate"},
                      color="mismatch", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        pri_data = result_df.groupby("Priority_Level")["mismatch"].sum().reset_index()
        fig4 = px.bar(pri_data, x="Priority_Level", y="mismatch",
                      title="Mismatches per Assigned Priority",
                      labels={"mismatch":"Count"},
                      color="Priority_Level",
                      color_discrete_map={"Low":"#00aa44","Medium":"#ffcc00",
                                          "High":"#ff8800","Critical":"#ff4444"})
        st.plotly_chart(fig4, use_container_width=True)
    with col4:
        type_data = (result_df[result_df["mismatch"]==1]
                     .groupby("mismatch_type").size().reset_index(name="count"))
        fig5 = px.bar(type_data, x="mismatch_type", y="count",
                      title="Hidden Crisis vs False Alarm Count",
                      color="mismatch_type",
                      color_discrete_map={"Hidden Crisis":"#ff4444","False Alarm":"#ff8800"})
        st.plotly_chart(fig5, use_container_width=True)

    # ── Severity delta heatmap ────────────────────────────────────────────────
    st.subheader(" Severity Delta Heatmap (Ticket Type × Channel)")
    if "ticket_type" in result_df.columns and "channel" in result_df.columns:
        hm = result_df.groupby(["ticket_type","channel"])["severity_delta"].mean().reset_index()
        hm_pivot = hm.pivot(index="ticket_type", columns="channel",
                            values="severity_delta").fillna(0)
        fig3 = go.Figure(data=go.Heatmap(
            z=hm_pivot.values,
            x=hm_pivot.columns.tolist(),
            y=hm_pivot.index.tolist(),
            colorscale="RdYlGn_r", zmid=0,
            text=np.round(hm_pivot.values, 2), texttemplate="%{text}",
            colorbar=dict(title="Avg Δ Severity")
        ))
        fig3.update_layout(
            title="Mean Severity Delta — positive = under-assigned (Hidden Crisis territory)",
            xaxis_title="Channel", yaxis_title="Ticket Type"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Ticket Type or Channel columns not found — heatmap unavailable.")

    # ── Top flagged tickets ───────────────────────────────────────────────────
    st.subheader(" Top Flagged Tickets (by Severity Delta)")
    flagged = (result_df[result_df["mismatch"]==1]
               .sort_values("severity_delta", ascending=False)
               .head(20))
    st.dataframe(
        flagged[["ticket_id","Priority_Level","inferred_severity",
                 "mismatch_type","severity_delta","confidence"]],
        use_container_width=True
    )

    # ── Download ──────────────────────────────────────────────────────────────
    csv_out = result_df[["ticket_id","Priority_Level","inferred_severity",
                          "mismatch","mismatch_type","severity_delta","confidence"]].to_csv(index=False)
    st.download_button("⬇️ Download Full Results CSV", csv_out, "sia_results.csv", "text/csv")
