"""
predict.py — Support Integrity Auditor (SIA) Inference
MARS Open Projects 2026

Usage:
    python predict.py --input tickets.csv --model sia_model_final --output results/

Outputs:
    results/predictions.csv
    results/dossiers.json
"""

import os, re, json, argparse, warnings
import numpy as np
import pandas as pd
from scipy.special import softmax as sp_softmax
from tqdm import tqdm
import spacy
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

warnings.filterwarnings("ignore")
tqdm.pandas()

# ── Constants (must match training) ─────────────────────────────────────────
PRIORITY_ORDER  = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
PRIORITY_LABELS = {v: k for k, v in PRIORITY_ORDER.items()}
MAX_LEN         = 256

CRITICAL_KEYWORDS  = ["urgent","critical","emergency","immediately","asap","broken","outage",
                       "down","crash","failure","data loss","security breach","cannot access",
                       "system down","not working","blocked","escalate","unacceptable","lawsuit",
                       "legal","refund","fraud","hacked"]
HIGH_KEYWORDS      = ["issue","problem","error","bug","slow","delay","frustrat","not responding",
                       "failed","incorrect","wrong","missing","stuck"]
LOW_KEYWORDS       = ["question","inquiry","wondering","curious","how do i",
                       "feature request","suggestion","feedback","would like","could you"]
ESCALATION_PHRASES = ["speak to manager","speak to supervisor","escalate","not acceptable",
                       "very disappointed","extremely frustrated","worst experience"]
NEGATION_WORDS     = ["not","n't","never","no","unable","cannot","can't"]

nlp = spacy.load("en_core_web_sm")

# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_resolution_time(val) -> float:
    if pd.isna(val): return np.nan
    val  = str(val)
    nums = re.findall(r"[\d\.]+", val)
    if not nums: return np.nan
    h = float(nums[0])
    if "day"  in val.lower(): h *= 24
    if "week" in val.lower(): h *= 168
    return h

def rule_based_severity(text: str) -> float:
    t      = text.lower()
    doc    = nlp(t[:2000])
    tokens = [tok.text for tok in doc]
    score  = 1.0
    score += min(sum(1 for kw in CRITICAL_KEYWORDS  if kw in t) * 0.6, 2.0)
    score += min(sum(1 for kw in HIGH_KEYWORDS       if kw in t) * 0.2, 0.8)
    score -= min(sum(1 for kw in LOW_KEYWORDS         if kw in t) * 0.3, 0.9)
    score += sum(1 for ph in ESCALATION_PHRASES if ph in t) * 0.5
    score += min(sum(1 for tok in tokens if tok in NEGATION_WORDS) * 0.15, 0.5)
    score += min(text.count("!") * 0.1, 0.3)
    score += min(len(re.findall(r"\b[A-Z]{3,}\b", text)) * 0.15, 0.45)
    return float(np.clip(score, 0, 3))

def build_input_text(row) -> str:
    chan  = str(row.get("channel",     "unknown"))
    ttype = str(row.get("ticket_type", "unknown"))
    rt    = row.get("resolution_hours", np.nan)
    rt_s  = f"{rt:.1f}h" if not (rt is None or (isinstance(rt, float) and np.isnan(rt))) else "unknown"
    return f"[CHANNEL: {chan}] [TYPE: {ttype}] [RT: {rt_s}] {row['text']}"

# ── Evidence & Dossier ───────────────────────────────────────────────────────
def extract_evidence(row, rt_mean: float, rt_std: float) -> list:
    text      = str(row.get("text", ""))
    evidence  = []
    crit_hits = [kw for kw in CRITICAL_KEYWORDS  if kw in text.lower()]
    high_hits = [kw for kw in HIGH_KEYWORDS       if kw in text.lower()]
    esc_hits  = [ph for ph in ESCALATION_PHRASES  if ph in text.lower()]

    if crit_hits:
        evidence.append({"signal":"critical_keyword","value":", ".join(crit_hits[:5]),
                         "source_field":"description","weight":f"{len(crit_hits)*0.6:.2f}"})
    if high_hits:
        evidence.append({"signal":"high_severity_keyword","value":", ".join(high_hits[:5]),
                         "source_field":"description","weight":f"{len(high_hits)*0.2:.2f}"})
    if esc_hits:
        evidence.append({"signal":"escalation_phrase","value":", ".join(esc_hits[:3]),
                         "source_field":"description","weight":"0.50"})
    excl = text.count("!")
    if excl > 0:
        evidence.append({"signal":"exclamation_density","value":str(excl),
                         "source_field":"description","weight":f"{min(excl*0.1,0.3):.2f}"})
    rt = row.get("resolution_hours", np.nan)
    if rt is not None and not (isinstance(rt, float) and np.isnan(rt)):
        high_rt = rt > (rt_mean + rt_std)
        evidence.append({"signal":"resolution_time","value":f"{rt:.1f} hours",
                         "source_field":"resolution_time",
                         "interpretation":"Abnormally high — underlying complexity" if high_rt else "Within normal range"})
    chan = row.get("channel", None)
    if chan and str(chan).lower() in ["phone","social_media","social media"]:
        evidence.append({"signal":"channel","value":str(chan),
                         "source_field":"channel","weight":"0.30"})
    if not evidence:
        evidence.append({"signal":"text_semantics","value":"semantic embedding cluster",
                         "source_field":"description","weight":"0.30"})
    return evidence

def generate_dossier(row, confidence: float, rt_mean: float, rt_std: float) -> dict:
    evidence = extract_evidence(row, rt_mean, rt_std)
    mtype    = row["mismatch_type"]
    assigned = row["priority"]
    inferred = row["inferred_severity"]
    sigs     = [e["signal"] for e in evidence]
    if mtype == "Hidden Crisis":
        analysis = (f"Ticket assigned {assigned} but signals indicate true severity of {inferred}. "
                    f"Evidence from {', '.join(sigs[:3])} is traceable to input fields. "
                    f"Immediate re-triage is recommended to prevent SLA breach.")
    else:
        analysis = (f"Ticket inflated to {assigned}; inferred severity is {inferred}. "
                    f"Evidence from {', '.join(sigs[:3])} is traceable to input fields. "
                    f"Deprioritisation advised to free bandwidth for genuine high-severity tickets.")
    return {
        "ticket_id"          : str(row.get("ticket_id","unknown")),
        "assigned_priority"  : assigned,
        "inferred_severity"  : inferred,
        "mismatch_type"      : mtype,
        "severity_delta"     : int(row["inferred_severity_num"] - row["priority_num"]),
        "feature_evidence"   : evidence,
        "constraint_analysis": analysis,
        "confidence"         : f"{confidence:.3f}"
    }

# ── Main ─────────────────────────────────────────────────────────────────────
def predict(input_csv: str, model_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df.rename(columns={"Ticket_ID":"ticket_id","Ticket_Subject":"subject",
                        "Ticket_Description":"description","Ticket_Priority":"priority",
                        "Ticket_Channel":"channel","Ticket_Type":"ticket_type",
                        "Resolution_Time":"resolution_time"}, inplace=True)
    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    df["text"] = (df.get("subject","").fillna("") + " [SEP] " + df["description"].fillna("")).str.strip()
    df["priority_num"] = df["priority"].map(PRIORITY_ORDER).fillna(1).astype(int)
    if "resolution_time" in df.columns:
        df["resolution_hours"] = df["resolution_time"].apply(parse_resolution_time)
    else:
        df["resolution_hours"] = np.nan

    rt_mean = df["resolution_hours"].mean()
    rt_std  = df["resolution_hours"].std()

    # Signal A
    print("Running NLP severity scoring ...")
    df["signal_nlp_score"]    = df["text"].progress_apply(rule_based_severity)
    df["signal_nlp_severity"] = pd.cut(df["signal_nlp_score"],
        bins=[-0.1,0.75,1.5,2.25,3.1], labels=[0,1,2,3]).astype(int)

    # Signal B
    print("Running semantic clustering ...")
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)
    km         = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(embeddings)
    csmap = (df.groupby("cluster")["signal_nlp_score"].median()
               .rank(method="first").sub(1).astype(int).to_dict())
    df["signal_cluster_severity"] = df["cluster"].map(csmap)
    df["signal_rt_severity"]      = df["signal_nlp_severity"]  # fallback

    # Fuse
    df["inferred_severity_score"] = (0.45 * df["signal_nlp_severity"] +
                                     0.30 * df["signal_cluster_severity"] +
                                     0.25 * df["signal_rt_severity"])
    df["inferred_severity_num"] = df["inferred_severity_score"].round().astype(int).clip(0, 3)
    df["inferred_severity"]     = df["inferred_severity_num"].map(PRIORITY_LABELS)
    df["mismatch_type"]         = df.apply(lambda r: (
        "Consistent" if r["inferred_severity_num"] == r["priority_num"]
        else ("Hidden Crisis" if r["inferred_severity_num"] > r["priority_num"] else "False Alarm")), axis=1)

    # Model inference
    print("Running classifier ...")
    df["model_input"] = df.apply(build_input_text, axis=1)
    tokenizer         = AutoTokenizer.from_pretrained(model_dir)
    base_model        = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-small", num_labels=2, ignore_mismatched_sizes=True)
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval().to(device)

    preds, probs = [], []
    for i in tqdm(range(0, len(df), 32), desc="Inference"):
        batch  = df["model_input"].iloc[i:i+32].tolist()
        enc    = tokenizer(batch, truncation=True, max_length=MAX_LEN,
                           padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits.cpu().numpy()
        p = sp_softmax(logits, axis=-1)
        preds.extend(np.argmax(p, axis=-1).tolist())
        probs.extend(p[:, 1].tolist())

    df["predicted_mismatch"] = preds
    df["confidence"]         = probs

    # Dossiers for predicted mismatches
    dossiers = []
    for _, row in df[df["predicted_mismatch"] == 1].iterrows():
        dossiers.append(generate_dossier(row, row["confidence"], rt_mean, rt_std))

    # Save
    out_cols = ["ticket_id","priority","inferred_severity","predicted_mismatch","mismatch_type","confidence"]
    df[out_cols].to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    with open(os.path.join(output_dir, "dossiers.json"), "w") as f:
        json.dump(dossiers, f, indent=2)
    print(f"\nSaved predictions → {output_dir}/predictions.csv")
    print(f"Saved dossiers   → {output_dir}/dossiers.json")
    print(f"Flagged mismatches: {df['predicted_mismatch'].sum()} / {len(df)}")
    return df, dossiers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True,               help="Input CSV path")
    parser.add_argument("--model",  default="sia_model_final",   help="Model directory")
    parser.add_argument("--output", default="results",           help="Output directory")
    args = parser.parse_args()
    predict(args.input, args.model, args.output)
