"""
train_pipeline.py — Support Integrity Auditor (SIA)
MARS Open Projects 2026

Usage:
    python train_pipeline.py --data customer_support_tickets.csv
"""

import os, re, json, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import spacy
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

warnings.filterwarnings("ignore")

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ── Constants ────────────────────────────────────────────────────────────────
PRIORITY_ORDER  = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
PRIORITY_LABELS = {v: k for k, v in PRIORITY_ORDER.items()}
SEED            = 42
MAX_LEN         = 256
BATCH_SIZE      = 16
EPOCHS          = 4

CRITICAL_KEYWORDS   = ["urgent","critical","emergency","immediately","asap","broken","outage",
                        "down","crash","failure","data loss","security breach","cannot access",
                        "system down","not working","blocked","escalate","unacceptable","lawsuit",
                        "legal","refund","fraud","hacked"]
HIGH_KEYWORDS       = ["issue","problem","error","bug","slow","delay","frustrat","not responding",
                        "failed","incorrect","wrong","missing","stuck"]
LOW_KEYWORDS        = ["question","inquiry","wondering","curious","when","how do i",
                        "feature request","suggestion","feedback","would like","could you"]
ESCALATION_PHRASES  = ["speak to manager","speak to supervisor","escalate","not acceptable",
                        "very disappointed","extremely frustrated","worst experience"]
NEGATION_WORDS      = ["not","n't","never","no","unable","cannot","can't"]

# ── NLP ──────────────────────────────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

def rule_based_severity(text: str) -> float:
    t   = text.lower()
    doc = nlp(t[:2000])
    tokens = [tok.text for tok in doc]
    score   = 1.0
    score  += min(sum(1 for kw in CRITICAL_KEYWORDS  if kw in t) * 0.6, 2.0)
    score  += min(sum(1 for kw in HIGH_KEYWORDS       if kw in t) * 0.2, 0.8)
    score  -= min(sum(1 for kw in LOW_KEYWORDS         if kw in t) * 0.3, 0.9)
    score  += sum(1 for ph in ESCALATION_PHRASES if ph in t) * 0.5
    score  += min(sum(1 for tok in tokens if tok in NEGATION_WORDS) * 0.15, 0.5)
    score  += min(text.count("!") * 0.1, 0.3)
    score  += min(len(re.findall(r"\b[A-Z]{3,}\b", text)) * 0.15, 0.45)
    return float(np.clip(score, 0, 3))

def parse_resolution_time(val) -> float:
    if pd.isna(val): return np.nan
    val  = str(val)
    nums = re.findall(r"[\d\.]+", val)
    if not nums: return np.nan
    h = float(nums[0])
    if "day"  in val.lower(): h *= 24
    if "week" in val.lower(): h *= 168
    return h

def build_input_text(row, rt_mean: float) -> str:
    chan  = row.get("channel",     "unknown")
    ttype = row.get("ticket_type", "unknown")
    rt    = row.get("resolution_hours", np.nan)
    rt_s  = f"{rt:.1f}h" if not pd.isna(rt) else "unknown"
    return f"[CHANNEL: {chan}] [TYPE: {ttype}] [RT: {rt_s}] {row['text']}"

# ── Data Loading ─────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df.rename(columns={
    "Ticket_ID": "ticket_id",
    "Ticket_Subject": "subject",
    "Ticket_Description": "description",
    "Priority_Level": "priority",
    "Ticket_Channel": "channel",
    "Customer_Email": "customer_email"
}, inplace=True)
    df.dropna(subset=["priority", "description"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    df["text"] = (df.get("subject", "").fillna("") + " [SEP] " + df["description"]).str.strip()
    df["priority_num"] = df["priority"].map(PRIORITY_ORDER)
    df.dropna(subset=["priority_num"], inplace=True)
    df["priority_num"]    = df["priority_num"].astype(int)
    if "resolution_time" in df.columns:
        df["resolution_hours"] = df["resolution_time"].apply(parse_resolution_time)
    else:
        df["resolution_hours"] = np.nan
    print(f"Loaded {len(df)} tickets.")
    return df

# ── Pseudo-Labelling ─────────────────────────────────────────────────────────
def generate_pseudo_labels(df: pd.DataFrame):
    print("Signal A: Rule-based NLP ...")
    tqdm.pandas()
    df["signal_nlp_score"]    = df["text"].progress_apply(rule_based_severity)
    df["signal_nlp_severity"] = pd.cut(df["signal_nlp_score"],
        bins=[-0.1, 0.75, 1.5, 2.25, 3.1], labels=[0, 1, 2, 3]).astype(int)

    print("Signal B: Sentence embeddings + KMeans ...")
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(df["text"].tolist(), batch_size=64, show_progress_bar=True)
    df["embedding"] = list(embeddings)
    km             = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    df["cluster"]  = km.fit_predict(embeddings)
    csmap = (df.groupby("cluster")["signal_nlp_score"].median()
               .rank(method="first").sub(1).astype(int).to_dict())
    df["signal_cluster_severity"] = df["cluster"].map(csmap)

    print("Signal C: Resolution-time regression ...")
    valid = df["resolution_hours"].notna()
    if valid.sum() > 100:
        rt_model = LinearRegression()
        rt_model.fit(np.stack(df.loc[valid, "embedding"].values), df.loc[valid, "resolution_hours"].values)
        rt_pred = rt_model.predict(np.stack(df["embedding"].values))
        df["signal_rt_severity"] = pd.qcut(rt_pred, q=4, labels=[0,1,2,3], duplicates="drop").astype(int)
    else:
        df["signal_rt_severity"] = df["signal_nlp_severity"]

    # Fuse
    df["inferred_severity_score"] = (0.45 * df["signal_nlp_severity"] +
                                     0.30 * df["signal_cluster_severity"] +
                                     0.25 * df["signal_rt_severity"])
    df["inferred_severity_num"] = df["inferred_severity_score"].round().astype(int).clip(0, 3)
    df["inferred_severity"]     = df["inferred_severity_num"].map(PRIORITY_LABELS)
    df["mismatch"]              = (df["inferred_severity_num"] != df["priority_num"]).astype(int)
    df["mismatch_type"]         = df.apply(
        lambda r: ("Consistent" if r["mismatch"] == 0
                   else ("Hidden Crisis" if r["inferred_severity_num"] > r["priority_num"]
                         else "False Alarm")), axis=1)

    agree = (df["signal_nlp_severity"] == df["signal_cluster_severity"]).mean()
    print(f"Pseudo-label signal agreement (NLP vs Cluster): {agree:.3f}")
    print(df["mismatch"].value_counts().to_string())
    return df, embedder

# ── Classifier ───────────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss(weight=self.class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    rec   = recall_score(labels, preds, average=None, zero_division=0)
    return {
        "accuracy":          accuracy_score(labels, preds),
        "macro_f1":          f1_score(labels, preds, average="macro", zero_division=0),
        "recall_consistent": float(rec[0]) if len(rec) > 0 else 0.0,
        "recall_mismatch":   float(rec[1]) if len(rec) > 1 else 0.0,
    }

def train(df: pd.DataFrame, output_dir: str = "sia_model_final"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    rt_mean = df["resolution_hours"].mean()
    df["model_input"] = df.apply(lambda r: build_input_text(r, rt_mean), axis=1)

    train_df, test_df  = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["mismatch"])
    train_df, val_df   = train_test_split(train_df, test_size=0.12, random_state=SEED, stratify=train_df["mismatch"])
    print(f"Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")

    cw = compute_class_weight("balanced", classes=np.array([0,1]), y=train_df["mismatch"].values)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding=False)

    def to_ds(frame):
        return Dataset.from_dict({"text": frame["model_input"].tolist(),
                                  "labels": frame["mismatch"].tolist()}).map(tok, batched=True, remove_columns=["text"])

    train_ds, val_ds, test_ds = to_ds(train_df), to_ds(val_df), to_ds(test_df)

    base  = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-small", num_labels=2, ignore_mismatched_sizes=True)
    model = get_peft_model(base, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["query_proj","value_proj"], bias="none"))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="./sia_ckpts", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-4, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        fp16=(device == "cuda"), seed=SEED, report_to="none"
    )
    trainer = WeightedTrainer(
        class_weights=class_weights, model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Evaluate
    preds_out  = trainer.predict(test_ds)
    y_pred     = np.argmax(preds_out.predictions, axis=-1)
    y_true     = test_df["mismatch"].values
    print("\n=== TEST RESULTS ===")
    print(classification_report(y_true, y_pred, target_names=["Consistent","Mismatch"]))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    print(f"\n✓ Acc={acc:.3f} | MacroF1={mf1:.3f} | Recall={rec}")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    test_df.to_csv(os.path.join(output_dir, "test_set.csv"), index=False)
    np.save(os.path.join(output_dir, "test_preds.npy"), preds_out.predictions)
    print(f"Model saved → {output_dir}/")
    return trainer, tokenizer, test_df, preds_out

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="customer_support_tickets.csv")
    parser.add_argument("--output", default="sia_model_final")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df, embedder = generate_pseudo_labels(load_and_preprocess(args.data))
    train(df, args.output)
    print("\nTraining complete.")
