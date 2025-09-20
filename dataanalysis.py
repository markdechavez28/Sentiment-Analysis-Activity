from pathlib import Path
import pandas as pd
import numpy as np
import sys
import re
import traceback

# Transformers
try:
    from transformers import pipeline, AutoTokenizer
except Exception as e:
    print("error", file=sys.stderr)
    raise

INPUT_XLSX = Path("Dataset") / "STS Art Dataset - Cleaned.xlsx"
OUTPUT_DIR = Path("Dataset")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "sentiment_results.csv"
OUTPUT_XLSX = OUTPUT_DIR / "STS Art Dataset - With Sentiment - DOST.xlsx"

PRIMARY_MODEL = "dost-asti/RoBERTa-tl-sentiment-analysis"
FALLBACK_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
PIPELINE_DEVICE = -1   
BATCH_SIZE = 16
MIN_TEXT_LENGTH_TO_ANALYZE = 3

_TOKEN_RE = re.compile(r"\b\w[\w']*\b")

def safe_save_excel(df, path: Path, sheet_name="Data"):
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print("Saved:", path.resolve())
        return path
    except PermissionError:
        base = path.stem
        suffix = path.suffix
        parent = path.parent
        i = 1
        while True:
            alt = parent / f"{base} (copy {i}){suffix}"
            if not alt.exists():
                with pd.ExcelWriter(alt, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                print("Permission denied for original file. Saved to:", alt.resolve())
                return alt
            i += 1

def flatten_to_dicts(obj):
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        if len(obj) == 0:
            return []
        if all(isinstance(x, dict) for x in obj):
            return obj
        if isinstance(obj[0], list):
            for inner in obj:
                if isinstance(inner, list) and any(isinstance(it, dict) for it in inner):
                    return [it for it in inner if isinstance(it, dict)]
            flat = []
            for inner in obj:
                if isinstance(inner, list):
                    flat.extend([it for it in inner if isinstance(it, dict)])
            return flat
        dicts = [x for x in obj if isinstance(x, dict)]
        return dicts
    return []

def compute_compound_from_pipe_output(pipe_output):
    try:
        dicts = flatten_to_dicts(pipe_output)
        if not dicts:
            return 0.0, 0.0, 1.0, 0.0

        labels_scores = {}
        for d in dicts:
            lab = str(d.get("label", "")).upper()
            sc = float(d.get("score", 0.0))
            labels_scores[lab] = sc

        pos_score = neg_score = neu_score = 0.0
        for k, v in labels_scores.items():
            kk = k.upper()
            if "POS" in kk or "POSITIVE" in kk or kk == "LABEL_2":
                pos_score += v
            elif "NEG" in kk or "NEGATIVE" in kk or kk == "LABEL_0":
                neg_score += v
            elif "NEU" in kk or "NEUTRAL" in kk or kk == "LABEL_1":
                neu_score += v
            else:
                if "POS" in kk:
                    pos_score += v
                elif "NEG" in kk:
                    neg_score += v
                elif "NEU" in kk:
                    neu_score += v
                else:
                    pass

        if pos_score + neg_score + neu_score == 0.0 and len(labels_scores) == 2:
            items = list(labels_scores.items())
            neg_score = items[0][1]; pos_score = items[1][1]
        if pos_score + neg_score + neu_score == 0.0 and len(labels_scores) > 0:
            best_label = max(labels_scores.items(), key=lambda x: x[1])[0]
            if "NEG" in best_label:
                neg_score = labels_scores[best_label]
            elif "POS" in best_label:
                pos_score = labels_scores[best_label]
            else:
                neu_score = labels_scores[best_label]

        compound = pos_score - neg_score
        total = pos_score + neg_score + neu_score
        if total > 1.0:
            pos_score /= total; neg_score /= total; neu_score /= total
        elif total < 1.0 and len(labels_scores) > 0:
            neu_score += max(0.0, 1.0 - total)

        return float(compound), float(max(0.0, pos_score)), float(max(0.0, neu_score)), float(max(0.0, neg_score))
    except Exception:
        print("Warning: unexpected pipeline output shape — falling back to neutral for this sample.", file=sys.stderr)
        traceback.print_exc()
        return 0.0, 0.0, 1.0, 0.0

def sentiment_category(c):
    if c <= -0.5:
        return "Strong Negative"
    elif c <= -0.05:
        return "Negative"
    elif c < 0.05:
        return "Neutral"
    elif c < 0.5:
        return "Positive"
    else:
        return "Strong Positive"

if not INPUT_XLSX.exists():
    print(f"ERROR: Input file not found at {INPUT_XLSX.resolve()}", file=sys.stderr)
    sys.exit(1)

xls = pd.read_excel(INPUT_XLSX, sheet_name=None, engine="openpyxl")
sheet_names = list(xls.keys())
sheet_name = sheet_names[0]
df = xls[sheet_name].copy()
print(f"Loaded '{sheet_name}' with {len(df)} rows.")

text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
def non_empty_str_count(series):
    return series.dropna().astype(str).str.strip().replace('', pd.NA).dropna().shape[0]

best_col = None
best_count = -1
for c in text_cols:
    cnt = non_empty_str_count(df[c])
    if cnt > best_count:
        best_count = cnt
        best_col = c

if best_col is None:
    avg_len = {c: df[c].astype(str).fillna("").str.len().mean() for c in df.columns}
    best_col = max(avg_len, key=avg_len.get)
    print("No explicit text column detected; selected by avg length:", best_col)
else:
    print(f"Using text column: '{best_col}' ({best_count} non-empty strings).")

texts = df[best_col].astype(str).fillna("").tolist()
n = len(texts)

def make_pipeline(model_name):
    try:
        p = pipeline("sentiment-analysis", model=model_name, device=PIPELINE_DEVICE, return_all_scores=True)
        print("Loaded pipeline:", model_name)
        return p
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

pipe = make_pipeline(PRIMARY_MODEL)
used_model = PRIMARY_MODEL
if pipe is None:
    print("Primary failed; trying fallback...")
    pipe = make_pipeline(FALLBACK_MODEL)
    used_model = FALLBACK_MODEL if pipe is not None else None

if pipe is None:
    print("ERROR: Could not load any transformers model. Install transformers & ensure internet.", file=sys.stderr)
    sys.exit(1)

try:
    tokenizer = AutoTokenizer.from_pretrained(used_model, use_fast=True)
    model_max_len = getattr(tokenizer, "model_max_length", 512) or 512
    if model_max_len is None or model_max_len <= 0 or model_max_len > 4096:
        model_max_len = 512
    print("Tokenizer max length:", model_max_len)
except Exception as e:
    print("Tokenizer load failed; falling back to max_length=512.", e)
    tokenizer = None
    model_max_len = 512

compounds = []; pos_scores = []; neu_scores = []; neg_scores = []
print(f"Scoring {n} rows with model {used_model} (batch_size={BATCH_SIZE})")
for i in range(0, n, BATCH_SIZE):
    batch = texts[i:i+BATCH_SIZE]
    idx_map = []
    input_texts = []
    for pos, txt in enumerate(batch):
        if txt is None or len(str(txt).strip()) < MIN_TEXT_LENGTH_TO_ANALYZE:
            idx_map.append(None)
        else:
            idx_map.append(len(input_texts))
            input_texts.append(txt)

    if len(input_texts) > 0:
        try:
            pipe_out = pipe(input_texts, truncation=True, padding="max_length", max_length=model_max_len)
        except Exception as e:
            print("Batch pipeline error — falling back to per-sample. Error:", e, file=sys.stderr)
            pipe_out = []
            for txt in input_texts:
                try:
                    single = pipe(txt, truncation=True, padding="max_length", max_length=model_max_len)
                    pipe_out.append(single)
                except Exception as e2:
                    print("Single-sample pipeline error:", e2, file=sys.stderr)
                    pipe_out.append(None)
    else:
        pipe_out = []

    out_iter = iter(pipe_out)
    for mapping in idx_map:
        if mapping is None:
            compounds.append(0.0); pos_scores.append(0.0); neu_scores.append(1.0); neg_scores.append(0.0)
        else:
            out_for_text = next(out_iter, None)
            if out_for_text is None:
                compounds.append(0.0); pos_scores.append(0.0); neu_scores.append(1.0); neg_scores.append(0.0)
            else:
                c, p, neu, neg = compute_compound_from_pipe_output(out_for_text)
                compounds.append(c); pos_scores.append(p); neu_scores.append(neu); neg_scores.append(neg)

df["_sent_compound"] = pd.Series(compounds, index=df.index).astype(float)
df["_sent_pos"] = pd.Series(pos_scores, index=df.index).astype(float)
df["_sent_neu"] = pd.Series(neu_scores, index=df.index).astype(float)
df["_sent_neg"] = pd.Series(neg_scores, index=df.index).astype(float)
df["_sent_category"] = df["_sent_compound"].apply(sentiment_category)

total_rows = len(df)
counts = df["_sent_category"].value_counts().reindex(
    ["Strong Negative","Negative","Neutral","Positive","Strong Positive"], fill_value=0)
print("\nCounts:\n", counts.to_frame("count"))
print("\nPercent:\n", ((counts / total_rows) * 100).round(2).to_frame("percent"))

df.to_csv(OUTPUT_CSV, index=False)
print("Saved CSV:", OUTPUT_CSV.resolve())
safe_save_excel(df, OUTPUT_XLSX, sheet_name=sheet_name)
print("Sentiment analysis complete.")
