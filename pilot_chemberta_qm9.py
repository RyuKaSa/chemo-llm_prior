"""
SIMDeLL pilot — ChemBERTa-2 latent space on QM9.

Pipeline:
    1. Load QM9 (yairschiff/qm9), sample 5k with seed=42.
    2. Encode SMILES with DeepChem/ChemBERTa-77M-MTR (frozen).
    3. Mean-pool over non-padding tokens -> (N, 384).
    4. UMAP to 2D.
    5. Color by chemical property, plot, save figure.
    6. Report silhouette scores.

Cached intermediates land in outputs/. Re-running is cheap: only missing
steps are recomputed.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
import torch

# ---------- Reproducibility ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- Hyperparams ----------
N_SAMPLES = 5000
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
BATCH_SIZE = 64
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Functional-group classification (mutually exclusive, priority order)
# Priority matters: aromatic > nitrile > amide > carboxyl > ester > carbonyl > alcohol > amine > ether > alkyne > other
# The first matching pattern wins, so put specific patterns before generic ones.
# ============================================================
FUNCTIONAL_GROUPS = [
    ("Aromatic",         "c"),                                  # any aromatic atom
    ("Nitrile",          "C#N"),
    ("Amide",            "[NX3][CX3](=O)"),
    ("Carboxyl",         "[CX3](=O)[OX2H]"),
    ("Ester",            "[CX3](=O)[OX2][#6]"),
    ("Carbonyl",         "[CX3]=[OX1]"),                        # catches aldehyde/ketone after the above
    ("Alcohol",          "[OX2H]"),
    ("Amine",            "[NX3;H2,H1,H0;!$(N=*);!$(NC=O)]"),
    ("Ether",            "[OD2]([#6])[#6]"),
    ("Alkyne",           "C#C"),
    ("Other",            None),
]


def assign_functional_group(smiles: str) -> str:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "Other"
    for name, patt in FUNCTIONAL_GROUPS:
        if patt is None:
            return name
        if mol.HasSubstructMatch(Chem.MolFromSmarts(patt)):
            return name
    return "Other"


# ============================================================
# Pipeline steps
# ============================================================
def load_qm9_sample(n: int, seed: int) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("yairschiff/qm9", split="train")
    df = ds.shuffle(seed=seed).select(range(n)).to_pandas()
    return df


def embed_smiles(smiles_list):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), BATCH_SIZE):
            batch = smiles_list[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)
            out = model(**inputs).last_hidden_state  # (B, L, 384)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def reduce_umap(X):
    import umap
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED
    )
    return reducer.fit_transform(X)


# ============================================================
# Plotting
# ============================================================
def plot_figure(X2d: np.ndarray, df: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """Single-panel hero figure.

    - All groups plotted as colored scatter points (largest underneath).
    - Only the TOP 4 groups by count get a bold text label on their cluster.
    - All groups (up to N_LEGEND) appear in an external legend on the right.
    - The external legend creates the outward margin naturally.

    Returns df with the added 'fgroup' column.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke

    # Configurable knobs
    TOP_N_ANNOTATE = 7    # how many clusters get an on-figure text label
    N_LEGEND = 10          # how many entries to show in the external legend

    # Compute functional-group labels
    df = df.copy()
    df["fgroup"] = df["canonical_smiles"].apply(assign_functional_group)

    fg_present = [name for name, _ in FUNCTIONAL_GROUPS if (df["fgroup"] == name).any()]
    counts = {g: int((df["fgroup"] == g).sum()) for g in fg_present}
    order_desc = sorted(fg_present, key=lambda g: -counts[g])

    # Color per group, locked by the FUNCTIONAL_GROUPS order so colors stay
    # consistent across re-runs.
    cmap = plt.cm.tab10
    colors = {g: cmap(i % 10) for i, g in enumerate(fg_present)}

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points: largest groups first so they go underneath, rare groups
    # stay visible on top.
    handles = {}
    for g in order_desc:
        m = df["fgroup"] == g
        h = ax.scatter(
            X2d[m, 0], X2d[m, 1],
            color=colors[g],
            s=10, alpha=0.65, edgecolor="none",
            label=f"{g}  (n={counts[g]})",
        )
        handles[g] = h

    # On-figure bold labels for the top N clusters only (top of zorder so
    # they sit above the points).
    for g in order_desc[:TOP_N_ANNOTATE]:
        m = df["fgroup"] == g
        cx = float(np.median(X2d[m, 0]))
        cy = float(np.median(X2d[m, 1]))
        ax.text(
            cx, cy, g,
            fontsize=13, fontweight="bold",
            ha="center", va="center",
            color="black",
            zorder=10,
            path_effects=[withStroke(linewidth=4, foreground="white")],
        )

    # External legend on the right — this is what gives the outward margin.
    leg_groups = order_desc[:N_LEGEND]
    leg_handles = [handles[g] for g in leg_groups]
    leg_labels = [f"{g}  (n={counts[g]})" for g in leg_groups]
    leg = ax.legend(
        leg_handles, leg_labels,
        title="Functional group",
        loc="lower right",
        frameon=True,
        framealpha=0.92,
        edgecolor="lightgray",
        fancybox=True,
        fontsize=10,
        markerscale=2.2,
        borderpad=0.6,
    )
    leg.get_title().set_fontsize(11)
    leg.get_title().set_fontweight("bold")
    leg.set_zorder(20)

    # Clean axes
    ax.set_xlabel("UMAP-1", fontsize=11)
    ax.set_ylabel("UMAP-2", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Outward margin around the point cloud so top-4 labels near the edges
    # don't bump the axis. set_xlim/set_ylim must come AFTER ax.scatter and
    # we must NOT use bbox_inches='tight' on save (it would crop the margin).
    PAD = 0.06
    x_min, x_max = X2d[:, 0].min(), X2d[:, 0].max()
    y_min, y_max = X2d[:, 1].min(), X2d[:, 1].max()
    ax.set_xlim(x_min - PAD * (x_max - x_min), x_max + PAD * (x_max - x_min))
    ax.set_ylim(y_min - PAD * (y_max - y_min), y_max + PAD * (y_max - y_min))

    ax.set_title(
        "ChemBERTa-2 organises 5,000 QM9 molecules by functional group\n"
        "(frozen pre-trained model, no fine-tuning, mean-pooled embeddings, UMAP-2D)",
        fontsize=12.5, pad=14,
    )

    fig.tight_layout()
    fig.savefig(out_path + ".png", dpi=300)
    fig.savefig(out_path + ".pdf")
    plt.close(fig)
    return df


# ============================================================
# Main with step-level caching
# ============================================================
def main():
    df_path = os.path.join(OUT_DIR, "qm9_sample.parquet")
    emb_path = os.path.join(OUT_DIR, "embeddings.npy")
    umap_path = os.path.join(OUT_DIR, "umap_2d.npy")

    # [1] Sample
    if os.path.exists(df_path):
        print("[1/5] Loading cached QM9 sample...")
        df = pd.read_parquet(df_path)
    else:
        print("[1/5] Loading QM9 sample (from HF)...")
        df = load_qm9_sample(N_SAMPLES, SEED)
        df.to_parquet(df_path)

    # [2] Embed
    if os.path.exists(emb_path):
        print("[2/5] Loading cached embeddings...")
        X = np.load(emb_path)
    else:
        print("[2/5] Encoding with ChemBERTa-2...")
        X = embed_smiles(df["canonical_smiles"].tolist())
        np.save(emb_path, X)

    # [3] UMAP
    if os.path.exists(umap_path):
        print("[3/5] Loading cached UMAP projection...")
        X2d = np.load(umap_path)
    else:
        print("[3/5] UMAP projection...")
        X2d = reduce_umap(X)
        np.save(umap_path, X2d)

    # [4] Plot
    print("[4/5] Plotting figure...")
    df = plot_figure(X2d, df, os.path.join(OUT_DIR, "umap_chemberta_qm9"))

    # [5] Silhouette scores on the projection (sampled for speed)
    print("[5/5] Silhouette scores (on UMAP-2D)...")
    from sklearn.metrics import silhouette_score
    for col in ["ring_count", "fgroup"]:
        nuniq = df[col].nunique()
        if nuniq < 2:
            print(f"  silhouette({col}) = N/A (only {nuniq} class)")
            continue
        labels = df[col].astype("category").cat.codes.values
        score = silhouette_score(X2d, labels, sample_size=2000, random_state=SEED)
        print(f"  silhouette({col:<12}) = {score:+.4f}  ({nuniq} classes)")

    # Bonus: print functional group distribution
    print("\nFunctional-group breakdown:")
    print(df["fgroup"].value_counts().to_string())


if __name__ == "__main__":
    main()