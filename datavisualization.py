from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys

INPUT_CSV = Path("Dataset") / "Sentiment Results.csv"
OUTPUT_PNG = Path("Assets") / "Final Output.png"

TOTAL_COLUMNS = 20
GRID_ROWS = 10
FIGSIZE = (14, 6)
DPI = 300

CATEGORY_ORDER = ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]

CELL_LINEWIDTH = 0.45
CELL_LINECOLOR = "#e6e6e6" 

def build_red_shades_cmap():
    stops = [
        "#5C1013","#8B0000","#C21807","#E53935","#FF6F61"   
    ]
    return LinearSegmentedColormap.from_list("reds_linear_soft", stops, N=256)

def safe_save_fig(fig, out_path: Path, dpi=300):
    try:
        fig.savefig(out_path, dpi=dpi)
        print("Saved:", out_path.resolve())
        return out_path
    except PermissionError:
        base = out_path.stem
        suffix = out_path.suffix
        parent = out_path.parent
        i = 1
        while True:
            alt = parent / f"{base} (copy {i}){suffix}"
            if not alt.exists():
                fig.savefig(alt, dpi=dpi)
                print("Permission denied. Saved to:", alt.resolve())
                return alt
            i += 1

if not INPUT_CSV.exists():
    print(f"ERROR: Input CSV not found at {INPUT_CSV.resolve()}. Run sentimentanalysis.py first.", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)
if "_sent_category" not in df.columns:
    print("ERROR: '_sent_category' column not found in CSV; re-run sentimentanalysis.py.", file=sys.stderr)
    sys.exit(1)

category_counts = df["_sent_category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
total = int(category_counts.sum())
if total == 0:
    raise RuntimeError("No sentiment-labeled rows found in CSV.")
category_percent = (category_counts / total) * 100.0

cols_per_category = (category_percent / 100.0 * TOTAL_COLUMNS).round().astype(int)
diff = TOTAL_COLUMNS - cols_per_category.sum()
if diff != 0:
    add_idx = int(np.argmax(category_percent.values))
    cols_per_category.iloc[add_idx] += diff

col_category_labels = []
for cat, ncols in cols_per_category.items():
    col_category_labels.extend([cat] * int(ncols))
if len(col_category_labels) < TOTAL_COLUMNS:
    pad_with = category_counts.idxmax()
    col_category_labels.extend([pad_with] * (TOTAL_COLUMNS - len(col_category_labels)))
elif len(col_category_labels) > TOTAL_COLUMNS:
    col_category_labels = col_category_labels[:TOTAL_COLUMNS]
col_category_labels = np.array(col_category_labels)

cmap = build_red_shades_cmap()

category_target_intensity = {
    "Strong Negative": 0.00, "Negative": 0.25, "Neutral": 0.50, "Positive": 0.80, "Strong Positive": 0.80}

category_centers = {}
for cat in CATEGORY_ORDER:
    cols_idx = np.where(col_category_labels == cat)[0]
    if len(cols_idx) == 0:
        continue
    center = cols_idx.mean()
    norm_center = center / max(1, (TOTAL_COLUMNS - 1))
    category_centers[cat] = norm_center

xp = []
fp = []
for cat in CATEGORY_ORDER:
    if cat in category_centers:
        xp.append(category_centers[cat])
        fp.append(category_target_intensity[cat])

if len(xp) == 0:
    xp = [0.0, 1.0]
    fp = [0.5, 0.5]

xp = np.array(xp)
fp = np.array(fp)

col_positions = np.linspace(0.0, 1.0, TOTAL_COLUMNS)
col_intensities = np.interp(col_positions, xp, fp)

col_intensities = np.clip(col_intensities, 0.03, 0.98)

image = np.tile(col_intensities.reshape(1, -1), (GRID_ROWS, 1))

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.imshow(image, origin="lower", aspect="equal", cmap=cmap, interpolation="bilinear")

ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

for x in range(TOTAL_COLUMNS + 1):
    ax.axvline(x - 0.5, color=CELL_LINECOLOR, linewidth=CELL_LINEWIDTH, zorder=3)
for y in range(GRID_ROWS + 1):
    ax.axhline(y - 0.5, color=CELL_LINECOLOR, linewidth=CELL_LINEWIDTH, zorder=3)

centers = []
labels = []
start = 0
for cat, ncols in cols_per_category.items():
    n = int(ncols)
    if n == 0:
        continue
    center = start + n / 2.0 - 0.5
    centers.append(center)
    labels.append(f"{cat}\n{category_percent[cat]:.1f}%")
    start += n

ax.set_xticks(centers)
ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right", va="top")
ax.xaxis.set_label_position('bottom')
ax.xaxis.tick_bottom()

ax.set_title("Shades of Warriors: Mapping Happiness Through a Sentiment Analysis", fontsize=14)

plt.tight_layout()
safe_save_fig(fig, OUTPUT_PNG, dpi=DPI)
plt.close(fig)

print("Saved:", OUTPUT_PNG.resolve())
