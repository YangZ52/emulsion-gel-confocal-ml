# %%
"""
Real-data-only publication benchmark for emulsion gel confocal images.

Goal:
  Compare formulation-only and descriptor ML models for
  predicting measured G'_1Hz with no formulation leakage.

This file is notebook-style Python. Open in VS Code/Jupyter or copy cells.

Expected files:
  emulsion gel confocal.xlsx
  real confocal images of emulsion gels/picture*.tif

Important:
  The workflow uses color-based droplet segmentation for handcrafted descriptors.
"""

# %%
# 1. Install in notebook if needed:
# %pip install numpy pandas matplotlib pillow scikit-image scipy scikit-learn shap

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import math
import os
import shutil
import textwrap
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage as ndi

from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, remove_small_holes, disk, opening, closing, skeletonize
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, watershed, find_boundaries
from skimage.draw import disk as draw_disk, line as draw_line

from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

try:
    CODE_PATH = Path(__file__).resolve()
except NameError:
    CODE_PATH = Path.cwd() / "src" / "emulsion_gel_confocal_ml_publication.py"

try:
    display
except NameError:
    def display(obj):
        print(obj)

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# %%
# 2. Load metadata

PROJECT_ROOT = Path(os.environ.get("EMULSION_GEL_PROJECT_ROOT", Path.cwd())).resolve()
ROOT = Path(os.environ.get("EMULSION_GEL_ROOT", PROJECT_ROOT)).resolve()
EXCEL_PATH = Path(
    os.environ.get("EMULSION_GEL_EXCEL", ROOT / "data" / "raw" / "emulsion gel confocal.xlsx")
).resolve()
IMAGE_DIR = Path(
    os.environ.get("EMULSION_GEL_IMAGE_DIR", ROOT / "data" / "raw" / "real confocal images of emulsion gels")
).resolve()

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE = Path(os.environ.get("EMULSION_GEL_OUTPUT_DIR", ROOT / "outputs")).resolve()
OUTPUT_DIR = OUTPUT_BASE / f"run_{RUN_TIMESTAMP}"
FIG_DIR = OUTPUT_DIR / "publication_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("Saving outputs to:", OUTPUT_DIR)
print("Saving figures to:", FIG_DIR)

if CODE_PATH.exists():
    shutil.copy2(CODE_PATH, OUTPUT_DIR / CODE_PATH.name)
    print("Saved code copy to:", OUTPUT_DIR / CODE_PATH.name)


def save_fig(name: str, dpi: int = 600):
    """Save current Matplotlib figure as high-resolution TIFF for publication."""
    tif = FIG_DIR / f"{name}.tif"
    plt.savefig(tif, dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    print("saved figure:", tif)


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "axes.linewidth": 1.1,
})

df = pd.read_excel(EXCEL_PATH)
EXCLUDED_IMAGES = ["picture16.tif"]
before_exclusion_n = len(df)
df = df[~df["image_file"].astype(str).isin(EXCLUDED_IMAGES)].reset_index(drop=True)
print(f"Excluded images from all training/testing and figures: {EXCLUDED_IMAGES}")
print(f"Rows before exclusion: {before_exclusion_n}; after exclusion: {len(df)}")
df["image_path"] = df["image_file"].apply(lambda x: IMAGE_DIR / x)
df["log_Gp"] = np.log10(df["G'_1Hz"].astype(float))

BREAKING_STRESS_CANDIDATES = [
    "breaking stress",
    "Breaking stress",
    "breaking_stress",
    "Breaking_stress",
    "breaking stress_Pa",
    "Breaking stress_Pa",
    "breaking_stress_Pa",
    "Breaking_stress_Pa",
    "Breaking stress (Pa)",
    "fracture stress",
    "Fracture stress",
    "fracture_stress",
    "Fracture_stress",
]
BREAKING_STRESS_COL = next((col for col in BREAKING_STRESS_CANDIDATES if col in df.columns), None)
if BREAKING_STRESS_COL is None:
    stress_like_cols = [
        col for col in df.columns
        if "stress" in str(col).lower() and ("break" in str(col).lower() or "fracture" in str(col).lower())
    ]
    BREAKING_STRESS_COL = stress_like_cols[0] if stress_like_cols else None

if BREAKING_STRESS_COL is not None:
    df["log_breaking_stress"] = np.log10(df[BREAKING_STRESS_COL].astype(float))
    print("Using breaking stress column:", BREAKING_STRESS_COL)
else:
    print(
        "Breaking stress prediction will be skipped: no breaking-stress column found. "
        f"Checked: {BREAKING_STRESS_CANDIDATES}"
    )

CAT_COLS = ["protein type", "heat"]
NUM_COLS = ["protein conc_percentage", "oil vol_percentage", "pH", "NaCl_mM", "CaCl2_mM"]
GROUP_COLS = CAT_COLS + NUM_COLS
df["formulation_group"] = df[GROUP_COLS].astype(str).agg("|".join, axis=1)

print(df.shape)
print("unique formulation groups:", df["formulation_group"].nunique())
display(df.head())

# %%
# 3. Image loading and preprocessing

def load_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"))
    return arr


def crop_scale_bar(rgb: np.ndarray, crop_bottom: int = 135, crop_right: int = 0) -> np.ndarray:
    """
    Your images have a white scale bar/text near bottom-right.
    Cropping bottom area avoids segmentation using scale-bar text as signal.
    """
    h, w = rgb.shape[:2]
    return rgb[: h - crop_bottom, : w - crop_right]


row = df.iloc[0]
rgb0 = load_rgb(row["image_path"])
rgb0_crop = crop_scale_bar(rgb0)

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(rgb0)
ax[0].set_title("original")
ax[1].imshow(rgb0_crop)
ax[1].set_title("scale-bar cropped")
for a in ax:
    a.axis("off")
plt.tight_layout()
save_fig("figure_01_scale_bar_crop_example")
plt.show()

# %%
# 4. Color-based droplet segmentation for descriptors

def segment_red_droplets(
    rgb: np.ndarray,
    min_size: int = 70,
    smooth_sigma: float = 1.0,
    marker_percentile: float = 65.0,
):
    rgb = crop_scale_bar(rgb)
    rgb_f = rgb.astype(float)
    red, green, blue = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]

    # Red dominance suppresses white and green network.
    red_score = red - 0.55 * green - 0.35 * blue
    red_score -= red_score.min()
    red_score /= red_score.max() + 1e-8

    smooth = gaussian(red_score, sigma=smooth_sigma)
    thr = threshold_otsu(smooth)
    mask = smooth > thr
    mask = opening(mask, disk(2))
    mask = closing(mask, disk(3))
    mask = remove_small_holes(mask, area_threshold=80)
    mask = remove_small_objects(mask, min_size=min_size)

    if mask.sum() == 0:
        raise ValueError("Empty droplet mask.")

    dist = ndi.distance_transform_edt(mask)
    markers = label(dist > np.percentile(dist[mask], marker_percentile))
    labels = watershed(-dist, markers, mask=mask)
    labels = clear_border(labels)
    labels = label(labels > 0)
    return labels, red_score, mask, rgb


def _safe_otsu(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0 or np.nanmax(arr) <= np.nanmin(arr):
        return float(np.nanmean(arr)) if arr.size else 0.0
    return float(threshold_otsu(arr))


def box_counting_fractal_dimension(mask: np.ndarray) -> float:
    """Estimate binary protein-network fractal dimension by box counting."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    max_power = int(np.floor(np.log2(min(h, w))))
    sizes = 2 ** np.arange(max_power, 2, -1)
    counts, used_sizes = [], []
    for size in sizes:
        h_trim = (h // size) * size
        w_trim = (w // size) * size
        if h_trim == 0 or w_trim == 0:
            continue
        blocks = mask[:h_trim, :w_trim].reshape(h_trim // size, size, w_trim // size, size)
        count = np.count_nonzero(blocks.any(axis=(1, 3)))
        if count > 0:
            counts.append(count)
            used_sizes.append(size)
    if len(counts) < 2:
        return 0.0
    slope, _ = np.polyfit(np.log(1.0 / np.asarray(used_sizes)), np.log(np.asarray(counts)), 1)
    return float(np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0))


def segment_green_protein_phase(
    rgb: np.ndarray,
    oil_mask: np.ndarray,
    smooth_sigma: float = 1.2,
    hole_area_threshold: int = 80,
    min_size: int = 60,
    green_floor_percentile: float = 35.0,
):
    """Segment bright green protein phase while excluding oil and black voids."""
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)

    # Relative green index separates green-rich pixels; an intensity floor keeps
    # black water/void regions from being mislabeled as protein.
    protein_score = green / (red + green + blue + 1e-8)
    protein_score = protein_score - protein_score.min()
    protein_score = protein_score / (protein_score.max() - protein_score.min() + 1e-8)
    green_norm = (green - green.min()) / (green.max() - green.min() + 1e-8)

    protein_smooth = gaussian(protein_score, sigma=smooth_sigma)
    green_smooth = gaussian(green_norm, sigma=smooth_sigma)
    green_candidates = green_smooth[~oil_mask.astype(bool)]
    green_floor = (
        max(_safe_otsu(green_candidates), float(np.percentile(green_candidates, green_floor_percentile)))
        if green_candidates.size
        else _safe_otsu(green_smooth)
    )

    protein_mask = (protein_smooth > _safe_otsu(protein_smooth)) & (green_smooth > green_floor)
    protein_mask = protein_mask & ~oil_mask.astype(bool)
    protein_mask = opening(protein_mask, disk(1))
    protein_mask = closing(protein_mask, disk(2))
    protein_mask = remove_small_holes(protein_mask, area_threshold=hole_area_threshold)
    protein_mask = remove_small_objects(protein_mask, min_size=min_size)
    return protein_mask, protein_score, green_smooth


def protein_network_descriptors(rgb: np.ndarray, oil_mask: np.ndarray) -> dict:
    protein_mask, _, _ = segment_green_protein_phase(rgb, oil_mask)
    skel = skeletonize(protein_mask)
    neighbor_count = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skel
    skeleton_pixels = int(skel.sum())
    branchpoints = int(np.count_nonzero(skel & (neighbor_count >= 3)))
    return {
        "protein_area_fraction": float(protein_mask.mean()),
        "protein_connection_density": float(branchpoints / (skeleton_pixels + 1e-8)),
        "protein_fractal_dimension": box_counting_fractal_dimension(protein_mask),
    }


def show_oil_protein_segmentation(rgb_full: np.ndarray, title: str = "segmentation check"):
    labels, red_score, oil_mask, rgb = segment_red_droplets(rgb_full)
    protein_mask, protein_score, green_smooth = segment_green_protein_phase(rgb, oil_mask)

    oil_only = np.zeros_like(rgb)
    oil_only[:, :, 0] = rgb[:, :, 0]
    oil_only[~oil_mask] = 0

    protein_only = np.zeros_like(rgb)
    protein_only[:, :, 1] = rgb[:, :, 1]
    protein_only[~protein_mask] = 0

    oil_boundary = find_boundaries(labels, mode="outer")
    protein_boundary = find_boundaries(protein_mask, mode="outer")
    overlay = rgb.copy()
    overlay[oil_boundary] = [255, 255, 0]
    overlay[protein_boundary] = [0, 255, 255]

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    ax = ax.ravel()
    ax[0].imshow(rgb)
    ax[0].set_title("cropped RGB")
    ax[1].imshow(oil_only)
    ax[1].set_title("oil only")
    ax[2].imshow(protein_only)
    ax[2].set_title("protein only")
    ax[3].imshow(overlay)
    ax[3].set_title("oil yellow / protein cyan")
    ax[4].imshow(oil_mask, cmap="gray")
    ax[4].set_title("oil mask")
    ax[5].imshow(protein_score, cmap="viridis")
    ax[5].set_title("relative green score")
    ax[6].imshow(green_smooth, cmap="Greens")
    ax[6].set_title("green brightness")
    ax[7].imshow(protein_mask, cmap="gray")
    ax[7].set_title("protein mask")
    for a in ax:
        a.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    save_fig("figure_02b_oil_protein_segmentation_check")
    plt.show()
    return labels, oil_mask, protein_mask


labels0, red_score0, mask0, rgb0_seg = segment_red_droplets(rgb0)
print("droplets:", labels0.max())

boundaries0 = find_boundaries(labels0, mode="outer")
overlay0 = rgb0_seg.copy()
overlay0[boundaries0] = [255, 255, 0]

fig, ax = plt.subplots(1, 5, figsize=(20, 4))
ax[0].imshow(rgb0_seg)
ax[0].set_title("cropped")
ax[1].imshow(red_score0, cmap="magma")
ax[1].set_title("red score")
ax[2].imshow(mask0, cmap="gray")
ax[2].set_title("mask")
ax[3].imshow(labels0, cmap="nipy_spectral")
ax[3].set_title("labels")
ax[4].imshow(overlay0)
ax[4].set_title("segmentation overlay")
for a in ax:
    a.axis("off")
plt.tight_layout()
save_fig("figure_02_segmentation_workflow_example")
plt.show()

labels_qc, oil_mask_qc, protein_mask_qc = show_oil_protein_segmentation(
    rgb0,
    title=row["image_file"],
)

# %%
# 5. Formulation encoding

form_transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ]
)

X_form = form_transformer.fit_transform(df).astype(np.float32)
FORM_DIM = X_form.shape[1]
print("formulation dim:", FORM_DIM)

# %%
# 6. Graph construction

@dataclass
class GraphSample:
    x: np.ndarray
    edge_index: np.ndarray
    formulation: np.ndarray
    y: float
    sample_id: str
    group: str
    descriptors: dict


def zscore_node_features(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return ((x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + eps)).astype(np.float32)


def graph_from_image(row: pd.Series, formulation_vec: np.ndarray, edge_gap_factor: float = 0.75) -> GraphSample:
    rgb_full = load_rgb(row["image_path"])
    labels, _, oil_mask, rgb = segment_red_droplets(rgb_full)
    red = rgb[:, :, 0].astype(float)
    green = rgb[:, :, 1].astype(float)
    protein_desc = protein_network_descriptors(rgb, oil_mask)
    pixel_size_um = float(row.get("Pixel_size_um", 1.0))

    props = regionprops(labels, intensity_image=green)
    props = [p for p in props if p.area > 0]
    n = len(props)
    if n < 2:
        raise ValueError(f"{row['image_file']}: fewer than 2 droplets")

    centers = np.array([[p.centroid[1], p.centroid[0]] for p in props], dtype=float)
    areas = np.array([p.area for p in props], dtype=float)
    radii = np.sqrt(areas / np.pi)
    green_i = np.array([p.mean_intensity for p in props], dtype=float)
    eccentricity = np.array([p.eccentricity for p in props], dtype=float)
    solidity = np.array([p.solidity for p in props], dtype=float)

    red_i, shell_green = [], []
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        crop = labels[minr:maxr, minc:maxc] == p.label
        red_i.append(float(red[minr:maxr, minc:maxc][crop].mean()))
        shell = ndi.binary_dilation(crop, iterations=4) & ~ndi.binary_dilation(crop, iterations=1)
        if shell.sum() > 0:
            shell_green.append(float(green[minr:maxr, minc:maxc][shell].mean()))
        else:
            shell_green.append(float(green_i[len(shell_green)]))
    red_i = np.array(red_i)
    shell_green = np.array(shell_green)

    contacts = set()
    median_radius = float(np.median(radii))
    nearest_distance = np.zeros(n)

    for i in range(n):
        d = np.linalg.norm(centers - centers[i], axis=1)
        nearest_distance[i] = np.sort(d)[1]
        for j in range(i + 1, n):
            gap = d[j] - (radii[i] + radii[j])
            if gap < edge_gap_factor * median_radius:
                contacts.add((i, j))
                contacts.add((j, i))

    if len(contacts) == 0:
        for i in range(n):
            d = np.linalg.norm(centers - centers[i], axis=1)
            for j in np.argsort(d)[1:4]:
                contacts.add((i, int(j)))
                contacts.add((int(j), i))

    edge_index = np.array(sorted(contacts), dtype=np.int64).T
    degree = np.bincount(edge_index[0], minlength=n).astype(float)

    local_area = np.zeros(n)
    for i in range(n):
        d = np.linalg.norm(centers - centers[i], axis=1)
        local = d < 4.0 * median_radius
        local_area[i] = areas[local].sum()

    node_raw = np.column_stack([
        radii, areas, centers[:, 0], centers[:, 1],
        degree, local_area, nearest_distance,
        green_i, shell_green, red_i, eccentricity, solidity,
    ])

    x = zscore_node_features(node_raw)
    total_pixels = labels.shape[0] * labels.shape[1]
    undirected_edges = edge_index.shape[1] / 2
    image_area_um2 = total_pixels * (pixel_size_um ** 2)
    radii_um = radii * pixel_size_um
    nearest_distance_um = nearest_distance * pixel_size_um

    desc = {
        "n_droplets": float(n / image_area_um2 * 1000.0),
        "oil_area_fraction": float(areas.sum() / total_pixels),
        "mean_radius": float(radii_um.mean()),
        "radius_cv": float(radii.std() / (radii.mean() + 1e-8)),
        "mean_degree": float(degree.mean()),
        "max_degree": float(degree.max()),
        "edge_density": float(undirected_edges / max(n * (n - 1) / 2, 1)),
        "mean_nearest_distance": float(nearest_distance_um.mean()),
        "mean_green": float(green_i.mean()),
        "mean_shell_green": float(shell_green.mean()),
        "mean_eccentricity": float(eccentricity.mean()),
        "mean_solidity": float(solidity.mean()),
    }
    desc.update(protein_desc)

    return GraphSample(
        x=x,
        edge_index=edge_index,
        formulation=formulation_vec,
        y=np.float32(row["log_Gp"]),
        sample_id=row["image_file"],
        group=row["formulation_group"],
        descriptors=desc,
    )


real_graphs = []
failed = []
for i, row in df.iterrows():
    try:
        real_graphs.append(graph_from_image(row, X_form[i]))
    except Exception as e:
        failed.append((row["image_file"], repr(e)))

print("graphs:", len(real_graphs), "failed:", failed)

desc_df = pd.DataFrame([
    {"sample_id": g.sample_id, "log_Gp": float(g.y), "group": g.group, **g.descriptors}
    for g in real_graphs
])
display(desc_df.head())
desc_df.to_csv(OUTPUT_DIR / "handcrafted_graph_image_descriptors.csv", index=False)


def segmentation_layers_for_publication(row: pd.Series):
    rgb = load_rgb(row["image_path"])
    labels, _, oil_mask, cropped = segment_red_droplets(rgb)
    protein_mask, _, _ = segment_green_protein_phase(cropped, oil_mask)

    oil_color = np.array([230, 74, 25], dtype=np.uint8)
    protein_color = np.array([46, 204, 113], dtype=np.uint8)

    oil_only = np.full_like(cropped, 245)
    oil_only[oil_mask] = oil_color

    protein_only = np.full_like(cropped, 245)
    protein_only[protein_mask] = protein_color

    phase_overlay = cropped.astype(np.float32)
    color_layer = np.zeros_like(phase_overlay)
    color_layer[oil_mask] = oil_color
    color_layer[protein_mask] = protein_color
    phase_pixels = oil_mask | protein_mask
    phase_overlay[phase_pixels] = 0.45 * phase_overlay[phase_pixels] + 0.55 * color_layer[phase_pixels]
    phase_overlay = np.clip(phase_overlay, 0, 255).astype(np.uint8)
    boundaries = find_boundaries(labels, mode="outer") | find_boundaries(protein_mask, mode="outer")
    phase_overlay[boundaries] = [255, 255, 255]

    return cropped, oil_only, protein_only, phase_overlay, labels, oil_mask, protein_mask


def make_segmentation_qc_montage(sample_rows: pd.DataFrame, max_images: int = 18):
    """Save a larger montage showing oil/protein overlays across samples."""
    rows = sample_rows.head(max_images)
    n = len(rows)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, rows.iterrows()):
        _, _, _, overlay, labels, oil_mask, protein_mask = segmentation_layers_for_publication(row)
        ax.imshow(overlay)
        ax.set_title(
            f"{row['image_file']}\n"
            f"oil={oil_mask.mean():.2f}, protein={protein_mask.mean():.2f}, droplets={labels.max()}",
            fontsize=11,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    save_fig("figure_03_oil_protein_segmentation_qc_montage")
    plt.show()


def make_segmentation_workflow_examples(sample_rows: pd.DataFrame, max_examples: int = 6):
    """Save several clear examples with RGB, oil, protein, and overlay panels."""
    rows = sample_rows.head(max_examples)
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.7 * n), squeeze=False)

    for i, (_, row) in enumerate(rows.iterrows()):
        cropped, oil_only, protein_only, overlay, labels, oil_mask, protein_mask = segmentation_layers_for_publication(row)
        panels = [
            ("Confocal RGB", cropped),
            ("Oil droplets", oil_only),
            ("Green protein", protein_only),
            ("Oil + protein overlay", overlay),
        ]
        for j, (title, image) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(image)
            if i == 0:
                ax.set_title(title, fontsize=14)
            if j == 0:
                ax.set_ylabel(
                    f"{row['image_file']}\n"
                    f"oil={oil_mask.mean():.2f}, protein={protein_mask.mean():.2f}, n={labels.max()}",
                    fontsize=11,
                )
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    save_fig("figure_04_oil_protein_segmentation_workflow_examples")
    plt.show()


def select_diverse_segmentation_rows(max_images: int = 18) -> pd.DataFrame:
    """Choose images spanning oil fraction, droplet size, protein topology, and G'."""
    merged = df.merge(desc_df, left_on="image_file", right_on="sample_id", how="inner", suffixes=("", "_desc"))
    diversity_cols = [
        "log_Gp_desc",
        "oil_area_fraction",
        "mean_radius",
        "radius_cv",
        "mean_nearest_distance",
        "protein_connection_density",
        "protein_fractal_dimension",
    ]
    diversity_cols = [c for c in diversity_cols if c in merged.columns]
    if not diversity_cols:
        return merged.head(max_images)

    z = merged[diversity_cols].astype(float)
    z = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)
    merged["_diversity_score"] = np.sqrt((z ** 2).sum(axis=1))
    merged = merged.sort_values("_diversity_score", ascending=False)

    chosen = []
    used_groups = set()
    for _, row in merged.iterrows():
        if row["formulation_group"] in used_groups:
            continue
        chosen.append(row)
        used_groups.add(row["formulation_group"])
        if len(chosen) >= max_images:
            break

    if len(chosen) < max_images:
        chosen_ids = {r["image_file"] for r in chosen}
        for _, row in merged.iterrows():
            if row["image_file"] not in chosen_ids:
                chosen.append(row)
                chosen_ids.add(row["image_file"])
                if len(chosen) >= max_images:
                    break

    return pd.DataFrame(chosen).drop(columns=["_diversity_score"], errors="ignore")


segmentation_example_rows = select_diverse_segmentation_rows(max_images=18)
segmentation_example_rows.to_csv(OUTPUT_DIR / "selected_diverse_segmentation_examples.csv", index=False)
make_segmentation_qc_montage(segmentation_example_rows, max_images=18)
make_segmentation_workflow_examples(segmentation_example_rows, max_examples=6)

# %%
# 7. Publication sanity plot: descriptors vs G'

SHAP_CONSISTENT_DESC_COLS = [
    "n_droplets",
    "oil_area_fraction",
    "mean_radius",
    "radius_cv",
    "mean_nearest_distance",
    "mean_shell_green",
    "mean_eccentricity",
    "protein_connection_density",
    "protein_fractal_dimension",
]

SHAP_CONSISTENT_DESCRIPTOR_LABELS = {
    "n_droplets": "Droplet density\n(per 1000 um$^2$)",
    "oil_area_fraction": "Oil phase fraction",
    "mean_radius": "Mean droplet radius\n(um)",
    "radius_cv": "Droplet size polydispersity",
    "mean_nearest_distance": "Mean nearest-neighbor distance\n(um)",
    "mean_shell_green": "Interfacial protein intensity",
    "mean_eccentricity": "Droplet anisotropy",
    "protein_connection_density": "Protein network connectivity",
    "protein_fractal_dimension": "Protein network complexity",
}

def publication_descriptor_axis_label(col: str) -> str:
    return "\n".join(
        textwrap.wrap(
            SHAP_CONSISTENT_DESCRIPTOR_LABELS[col],
            width=28,
            break_long_words=False,
            replace_whitespace=False,
        )
    )

plot_cols = SHAP_CONSISTENT_DESC_COLS
ncols = 3
nrows = math.ceil(len(plot_cols) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.8 * nrows))
for ax, col in zip(axes.ravel(), plot_cols):
    ax.scatter(desc_df[col], 10 ** desc_df["log_Gp"], s=64, alpha=0.84, color="#2F5597", edgecolor="white", linewidth=0.7)
    ax.set_xlabel(publication_descriptor_axis_label(col), fontsize=16, fontweight="semibold", labelpad=8)
    ax.set_ylabel("G'$_{1Hz}$ (Pa)", fontsize=15, fontweight="semibold", labelpad=7)
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=12)
for ax in axes.ravel()[len(plot_cols):]:
    ax.axis("off")
plt.tight_layout()
save_fig("figure_05_descriptor_vs_rheology")
plt.show()

if BREAKING_STRESS_COL is not None:
    plot_cols = SHAP_CONSISTENT_DESC_COLS
    ncols = 3
    nrows = math.ceil(len(plot_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.8 * nrows))
    for ax, col in zip(axes.ravel(), plot_cols):
        aligned_for_plot = df.set_index("image_file").loc[desc_df["sample_id"]]
        ax.scatter(
            desc_df[col],
            aligned_for_plot[BREAKING_STRESS_COL].astype(float),
            s=64,
            alpha=0.84,
            color="#2F5597",
            edgecolor="white",
            linewidth=0.7,
        )
        ax.set_xlabel(publication_descriptor_axis_label(col), fontsize=16, fontweight="semibold", labelpad=8)
        ax.set_ylabel("Breaking stress (Pa)", fontsize=15, fontweight="semibold", labelpad=7)
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=12)
    for ax in axes.ravel()[len(plot_cols):]:
        ax.axis("off")
    plt.tight_layout()
    save_fig("figure_05b_descriptor_vs_breaking_stress")
    plt.show()

# %%
# 8. Shared split: stratified GroupKFold by formulation and protein type

groups = np.array([g.group for g in real_graphs])
unique_groups = np.unique(groups)
N_SPLITS = min(5, len(unique_groups))

aligned_for_split = df.set_index("image_file").loc[[g.sample_id for g in real_graphs]].reset_index()
protein_strata = aligned_for_split["protein type"].astype(str).values

splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
splits = list(splitter.split(np.arange(len(real_graphs)), y=protein_strata, groups=groups))
print("folds:", len(splits))
fold_rows = []
for k, (train_idx, test_idx) in enumerate(splits, 1):
    train_counts = pd.Series(protein_strata[train_idx]).value_counts().to_dict()
    test_counts = pd.Series(protein_strata[test_idx]).value_counts().to_dict()
    train_ratio = len(train_idx) / len(protein_strata)
    test_ratio = len(test_idx) / len(protein_strata)
    print(
        k,
        "train samples:", len(train_idx),
        "test samples:", len(test_idx),
        "train ratio:", f"{train_ratio:.2f}",
        "test ratio:", f"{test_ratio:.2f}",
        "test groups:", len(np.unique(groups[test_idx])),
        "train protein:", train_counts,
        "test protein:", test_counts,
    )
    fold_rows.append({
        "fold": k,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "train_groups": len(np.unique(groups[train_idx])),
        "test_groups": len(np.unique(groups[test_idx])),
        "train_protein_counts": train_counts,
        "test_protein_counts": test_counts,
    })

fold_balance_df = pd.DataFrame(fold_rows)
display(fold_balance_df)
fold_balance_df.to_csv(OUTPUT_DIR / "stratified_group_fold_balance.csv", index=False)

# Previous validation style for comparison: grouped by formulation but not
# stratified by protein type. This is useful to show whether protein-balanced
# folds affect model performance.
grouped_splits = list(GroupKFold(n_splits=N_SPLITS).split(np.arange(len(real_graphs)), groups=groups))
grouped_fold_rows = []
for k, (train_idx, test_idx) in enumerate(grouped_splits, 1):
    grouped_fold_rows.append({
        "fold": k,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "train_ratio": len(train_idx) / len(protein_strata),
        "test_ratio": len(test_idx) / len(protein_strata),
        "train_groups": len(np.unique(groups[train_idx])),
        "test_groups": len(np.unique(groups[test_idx])),
        "train_protein_counts": pd.Series(protein_strata[train_idx]).value_counts().to_dict(),
        "test_protein_counts": pd.Series(protein_strata[test_idx]).value_counts().to_dict(),
    })
grouped_fold_balance_df = pd.DataFrame(grouped_fold_rows)
display(grouped_fold_balance_df)
grouped_fold_balance_df.to_csv(OUTPUT_DIR / "non_stratified_group_fold_balance.csv", index=False)

# %%
# 9. Publication model benchmark:
#    formulation only, structure only, formulation + structure

aligned = df.set_index("image_file").loc[[g.sample_id for g in real_graphs]].reset_index()
y = aligned["log_Gp"].values.astype(float)
X_form_aligned = form_transformer.transform(aligned).astype(float)

GP_TARGET = {
    "key": "Gp",
    "log_col": "log_Gp",
    "true_log_col": "true_log_Gp",
    "pred_log_col": "pred_log_Gp",
    "true_value_col": "true_Gp",
    "pred_value_col": "pred_Gp",
    "value_label": "G'",
    "measured_label": "Measured G'$_{1Hz}$ (Pa)",
    "predicted_label": "Predicted G'$_{1Hz}$ (Pa)",
    "mean_log_label": "Mean log10 G'$_{1Hz}$ (Pa)",
    "ba_diff_label": "Predicted - measured log10 G'$_{1Hz}$ (Pa)",
    "bar_r2_label": r"Grouped-CV $R^2$ for log$_{10}$ G'$_{1Hz}$ (Pa)",
    "bar_rmse_label": r"Grouped-CV RMSE for log$_{10}$ G'$_{1Hz}$ (Pa)",
    "shap_xlabel": "Mean |SHAP value| for log10 G'$_{1Hz}$ (Pa)",
    "formulation_shap_title": "Formulation SHAP for G', protein type excluded",
    "descriptor_shap_title": "Structure descriptor SHAP importance for G'",
    "output_tag": "",
    "figure_offset": 0,
}

TARGET_CONFIGS = [GP_TARGET]
if BREAKING_STRESS_COL is not None:
    BREAKING_STRESS_TARGET = {
        "key": "breaking_stress",
        "log_col": "log_breaking_stress",
        "true_log_col": "true_log_breaking_stress",
        "pred_log_col": "pred_log_breaking_stress",
        "true_value_col": "true_breaking_stress",
        "pred_value_col": "pred_breaking_stress",
        "value_label": "breaking stress",
        "measured_label": "Measured breaking stress (Pa)",
        "predicted_label": "Predicted breaking stress (Pa)",
        "mean_log_label": "Mean log10 breaking stress (Pa)",
        "ba_diff_label": "Predicted - measured log10 breaking stress (Pa)",
        "bar_r2_label": r"Grouped-CV $R^2$ for log$_{10}$ breaking stress (Pa)",
        "bar_rmse_label": r"Grouped-CV RMSE for log$_{10}$ breaking stress (Pa)",
        "shap_xlabel": "Mean |SHAP value| for log10 breaking stress (Pa)",
        "formulation_shap_title": "Formulation SHAP for breaking stress, protein type excluded",
        "descriptor_shap_title": "Structure descriptor SHAP importance for breaking stress",
        "output_tag": "breaking_stress_",
        "figure_offset": 20,
    }
    TARGET_CONFIGS.append(BREAKING_STRESS_TARGET)

DESC_COLS = SHAP_CONSISTENT_DESC_COLS
DESCRIPTOR_PUBLICATION_LABELS = SHAP_CONSISTENT_DESCRIPTOR_LABELS

pd.DataFrame(
    [{"code_name": k, "publication_name": v} for k, v in DESCRIPTOR_PUBLICATION_LABELS.items()]
).to_csv(OUTPUT_DIR / "publication_descriptor_names.csv", index=False)

X_desc = desc_df[DESC_COLS].values.astype(float)
X_desc_form = np.hstack([X_desc, X_form_aligned])


def plot_descriptor_pca_publication():
    """PCA overview of standardized microstructural descriptors."""
    desc_scaled = StandardScaler().fit_transform(X_desc)
    pca = PCA(n_components=2, random_state=SEED)
    scores = pca.fit_transform(desc_scaled)
    pca_df = aligned.copy()
    pca_df["PC1"] = scores[:, 0]
    pca_df["PC2"] = scores[:, 1]
    pca_df["log10_Gp"] = pca_df["log_Gp"]
    if BREAKING_STRESS_COL is not None:
        pca_df["log10_breaking_stress"] = pca_df["log_breaking_stress"]

    pca_scores = pca_df[["image_file", "protein type", "heat", "PC1", "PC2", "log10_Gp"]].copy()
    if BREAKING_STRESS_COL is not None:
        pca_scores["log10_breaking_stress"] = pca_df["log10_breaking_stress"]
    pca_scores.to_csv(OUTPUT_DIR / "microstructural_descriptor_pca_scores.csv", index=False)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=[DESCRIPTOR_PUBLICATION_LABELS[c] for c in DESC_COLS],
        columns=["PC1_loading", "PC2_loading"],
    )
    loadings["loading_strength"] = np.sqrt(loadings["PC1_loading"] ** 2 + loadings["PC2_loading"] ** 2)
    loadings.to_csv(OUTPUT_DIR / "microstructural_descriptor_pca_loadings.csv")
    loadings.sort_values("loading_strength", ascending=False).to_csv(
        OUTPUT_DIR / "microstructural_descriptor_pca_top_loading_directions.csv"
    )

    ncols = 3 if BREAKING_STRESS_COL is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.3 * ncols, 5.1), constrained_layout=True)
    axes = np.atleast_1d(axes)

    marker_map = {"N": "o", "Y": "s", "No": "o", "Yes": "s"}
    protein_colors = {"FPI": "#4C78A8", "QPI": "#F58518"}
    for protein, protein_df in pca_df.groupby("protein type"):
        for heat, heat_df in protein_df.groupby("heat"):
            axes[0].scatter(
                heat_df["PC1"],
                heat_df["PC2"],
                s=74,
                marker=marker_map.get(str(heat), "o"),
                color=protein_colors.get(str(protein), "#6A6A6A"),
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
                label=f"{protein}, heat {heat}",
            )
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    axes[0].set_title("Protein type and heat", fontsize=15, pad=10)

    gp_scatter = axes[1].scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["log10_Gp"],
        cmap="viridis",
        s=74,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
    )
    axes[1].set_title("log$_{10}$ G'$_{1Hz}$ (Pa)", fontsize=15, pad=10)
    gp_cbar = fig.colorbar(gp_scatter, ax=axes[1], fraction=0.046, pad=0.04)
    gp_cbar.set_label("log$_{10}$ G'$_{1Hz}$ (Pa)", fontsize=12)
    gp_cbar.ax.tick_params(labelsize=11)

    if BREAKING_STRESS_COL is not None:
        stress_scatter = axes[2].scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            c=pca_df["log10_breaking_stress"],
            cmap="magma",
            s=74,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
        )
        axes[2].set_title("log$_{10}$ breaking stress (Pa)", fontsize=15, pad=10)
        stress_cbar = fig.colorbar(stress_scatter, ax=axes[2], fraction=0.046, pad=0.04)
        stress_cbar.set_label("log$_{10}$ breaking stress (Pa)", fontsize=12)
        stress_cbar.ax.tick_params(labelsize=11)

    xlabel = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
    ylabel = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.axhline(0, color="0.86", linewidth=0.9, zorder=0)
        ax.axvline(0, color="0.86", linewidth=0.9, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="0.92", linewidth=0.7)
        ax.set_axisbelow(True)

    save_fig("figure_05c_microstructural_descriptor_pca")
    plt.show()

    # Companion biplot: arrows explain what high/low PC values mean.
    top_loadings = loadings.sort_values("loading_strength", ascending=False).head(7)
    fig, ax = plt.subplots(figsize=(8.2, 6.7))
    ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        s=44,
        color="0.72",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.72,
        zorder=1,
    )

    score_span = max(
        pca_df["PC1"].max() - pca_df["PC1"].min(),
        pca_df["PC2"].max() - pca_df["PC2"].min(),
    )
    arrow_scale = score_span * 0.36
    for label, row in top_loadings.iterrows():
        x_end = row["PC1_loading"] * arrow_scale
        y_end = row["PC2_loading"] * arrow_scale
        ax.arrow(
            0,
            0,
            x_end,
            y_end,
            color="#C00000",
            width=0.006 * score_span,
            head_width=0.045 * score_span,
            length_includes_head=True,
            alpha=0.9,
            zorder=3,
        )
        ax.text(
            x_end * 1.08,
            y_end * 1.08,
            label,
            fontsize=10,
            color="#7A0000",
            ha="left" if x_end >= 0 else "right",
            va="center",
            zorder=4,
        )

    def _pc_direction_text(pc_col: str, sign: int) -> str:
        ranked = loadings.sort_values(pc_col, ascending=(sign < 0))
        selected = ranked.head(3).index.tolist()
        prefix = "Higher" if sign > 0 else "Lower"
        return prefix + " " + pc_col.replace("_loading", "") + ":\n" + "\n".join(selected)

    ax.text(
        0.99,
        0.98,
        _pc_direction_text("PC1_loading", 1),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="0.82", alpha=0.92, boxstyle="round,pad=0.35"),
    )
    ax.text(
        0.01,
        0.02,
        _pc_direction_text("PC1_loading", -1),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="0.82", alpha=0.92, boxstyle="round,pad=0.35"),
    )
    ax.text(
        0.01,
        0.98,
        _pc_direction_text("PC2_loading", 1),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="0.82", alpha=0.92, boxstyle="round,pad=0.35"),
    )
    ax.text(
        0.99,
        0.02,
        _pc_direction_text("PC2_loading", -1),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox=dict(facecolor="white", edgecolor="0.82", alpha=0.92, boxstyle="round,pad=0.35"),
    )

    ax.axhline(0, color="0.82", linewidth=1.0, zorder=0)
    ax.axvline(0, color="0.82", linewidth=1.0, zorder=0)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title("PCA loading biplot: descriptor directions", fontsize=16, pad=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(color="0.92", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig("figure_05d_microstructural_descriptor_pca_biplot")
    plt.show()


plot_descriptor_pca_publication()


# %%
# 9a. Simulation-guided structure generator using HOOMD-blue when available
#
# This section is deliberately used for mechanistic interpretation, not for
# supervised G' prediction. Simulated structures are processed with the same
# descriptor pipeline and overlaid onto the experimental descriptor/PCA space.

def _initial_positions_2d(n_particles: int, box_size: float, rng: np.random.Generator) -> np.ndarray:
    grid_n = int(np.ceil(np.sqrt(n_particles)))
    coords = np.linspace(-0.42 * box_size, 0.42 * box_size, grid_n)
    xx, yy = np.meshgrid(coords, coords)
    pos = np.column_stack([xx.ravel(), yy.ravel()])[:n_particles]
    pos += rng.normal(scale=0.12 * box_size / grid_n, size=pos.shape)
    return pos


def simulate_droplet_positions_hoomd(
    attraction_strength: float,
    n_particles: int = 72,
    box_size: float = 40.0,
    steps: int = 900,
    seed: int = SEED,
):
    """
    Simulate 2D droplet aggregation with HOOMD-blue if available.

    The attraction strength is mapped to the Lennard-Jones epsilon parameter.
    This is a coarse-grained structure generator; it is not used to predict an
    absolute rheological modulus.
    """
    try:
        import hoomd
        import hoomd.md
    except ImportError:
        return None, "hoomd_not_installed"

    try:
        rng = np.random.default_rng(seed)
        device = hoomd.device.CPU()
        sim = hoomd.Simulation(device=device, seed=seed)
        snapshot = hoomd.Snapshot()
        if snapshot.communicator.rank == 0:
            # Use a finite 3D periodic box and later project x/y into a
            # confocal-like image. A zero-thickness 2D box can trigger
            # "particle no longer in box" errors in HOOMD-blue.
            snapshot.configuration.box = [box_size, box_size, box_size, 0, 0, 0]
            snapshot.particles.N = n_particles
            snapshot.particles.types = ["droplet"]
            pos2d = _initial_positions_2d(n_particles, box_size, rng)
            z = rng.normal(scale=0.02, size=n_particles)
            snapshot.particles.position[:] = np.column_stack([pos2d, z])
            snapshot.particles.diameter[:] = 1.0
        sim.create_state_from_snapshot(snapshot)

        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        lj = hoomd.md.pair.LJ(nlist=nlist)
        # Keep epsilon modest for numerical stability. The rendered protein
        # bridge density below carries the weak-to-strong attraction contrast.
        lj.params[("droplet", "droplet")] = {"epsilon": 0.08 + 0.18 * attraction_strength, "sigma": 1.0}
        lj.r_cut[("droplet", "droplet")] = 2.2
        integrator = hoomd.md.Integrator(dt=0.0002)
        integrator.forces.append(lj)
        integrator.methods.append(hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=0.25))
        sim.operations.integrator = integrator
        sim.run(steps)

        snap = sim.state.get_snapshot()
        if snap.communicator.rank == 0:
            return np.asarray(snap.particles.position[:, :2]), "hoomd"
        return None, "hoomd_nonroot"
    except Exception as exc:
        print(f"HOOMD simulation failed for epsilon={attraction_strength}: {exc}")
        return None, "hoomd_failed"


def simulate_droplet_positions_numpy(
    attraction_strength: float,
    n_particles: int = 72,
    box_size: float = 40.0,
    steps: int = 450,
    seed: int = SEED,
) -> np.ndarray:
    """Fallback aggregation surrogate used only when HOOMD-blue is unavailable."""
    rng = np.random.default_rng(seed)
    pos = _initial_positions_2d(n_particles, box_size, rng)
    cutoff = 4.2
    for _ in range(steps):
        disp = np.zeros_like(pos)
        for i in range(n_particles):
            delta = pos - pos[i]
            dist = np.linalg.norm(delta, axis=1) + 1e-8
            near = (dist < cutoff) & (dist > 0)
            if np.any(near):
                # Attraction pulls neighbors into bridged clusters; repulsion
                # prevents total collapse into one point.
                unit = delta[near] / dist[near, None]
                attraction = attraction_strength * 0.0018 * unit.sum(axis=0)
                overlap = dist[near] < 1.05
                repulsion = -0.018 * unit[overlap].sum(axis=0) if np.any(overlap) else 0.0
                disp[i] += attraction + repulsion
        pos += disp + rng.normal(scale=0.012, size=pos.shape)
        pos = np.clip(pos, -0.46 * box_size, 0.46 * box_size)
    return pos


def render_synthetic_confocal_like_image(
    positions: np.ndarray,
    attraction_strength: float,
    image_size: int = 620,
    box_size: float = 40.0,
    droplet_radius_px: int = 9,
) -> np.ndarray:
    # Render a blue/cyan confocal-like synthetic image. The protein network is
    # cyan-blue for visual quality but retains green-channel intensity so the
    # same descriptor extraction code can segment it consistently.
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    vignette = ((xx - image_size / 2) ** 2 + (yy - image_size / 2) ** 2) / (image_size / 2) ** 2
    rgb = np.zeros((image_size, image_size, 3), dtype=np.float32)
    rgb[:, :, 2] = 12 + 18 * np.clip(1 - vignette, 0, 1)
    rgb[:, :, 1] = 4 + 8 * np.clip(1 - vignette, 0, 1)
    scale = image_size * 0.80 / box_size
    center = image_size / 2
    pix = np.column_stack([center + positions[:, 0] * scale, center + positions[:, 1] * scale])

    # Protein bridges are drawn between close droplets, increasing with the
    # attraction parameter to mimic weak-to-strong network formation.
    cutoff_px = (2.6 + 0.35 * attraction_strength) * scale
    for i in range(len(pix)):
        delta = pix - pix[i]
        dist = np.linalg.norm(delta, axis=1)
        neighbors = np.where((dist > 0) & (dist < cutoff_px))[0]
        for j in neighbors:
            if j <= i:
                continue
            rr, cc = draw_line(int(pix[i, 1]), int(pix[i, 0]), int(pix[j, 1]), int(pix[j, 0]))
            valid = (rr >= 0) & (rr < image_size) & (cc >= 0) & (cc < image_size)
            bridge_strength = 120 + 32 * attraction_strength
            rgb[rr[valid], cc[valid], 1] = np.maximum(rgb[rr[valid], cc[valid], 1], bridge_strength)
            rgb[rr[valid], cc[valid], 2] = np.maximum(rgb[rr[valid], cc[valid], 2], 210 + 8 * attraction_strength)

    rgb[:, :, 1] = gaussian(rgb[:, :, 1], sigma=1.45, preserve_range=True)
    rgb[:, :, 2] = gaussian(rgb[:, :, 2], sigma=1.55, preserve_range=True)
    for x, y in pix:
        rr, cc = draw_disk((int(y), int(x)), radius=droplet_radius_px, shape=rgb.shape[:2])
        rgb[rr, cc, 0] = 238
        rgb[rr, cc, 1] = np.maximum(rgb[rr, cc, 1], 48)
        rgb[rr, cc, 2] = np.maximum(rgb[rr, cc, 2], 32)

        rr_core, cc_core = draw_disk((int(y), int(x)), radius=max(2, droplet_radius_px - 4), shape=rgb.shape[:2])
        rgb[rr_core, cc_core, 0] = 255
        rgb[rr_core, cc_core, 1] = np.maximum(rgb[rr_core, cc_core, 1], 78)
        rgb[rr_core, cc_core, 2] = np.maximum(rgb[rr_core, cc_core, 2], 58)

    rgb += np.random.default_rng(SEED).normal(scale=2.4, size=rgb.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def descriptors_from_synthetic_rgb(rgb: np.ndarray, pixel_size_um: float | None = None) -> dict:
    if pixel_size_um is None:
        pixel_size_um = float(df["Pixel_size_um"].median())
    labels, _, oil_mask, cropped = segment_red_droplets(rgb, min_size=35)
    protein_desc = protein_network_descriptors(cropped, oil_mask)
    props = regionprops(labels, intensity_image=cropped[:, :, 1].astype(float))
    props = [p for p in props if p.area > 15]
    if len(props) < 2:
        return {col: np.nan for col in DESC_COLS}

    centers = np.array([p.centroid for p in props], dtype=float)
    areas = np.array([p.area for p in props], dtype=float)
    radii = np.sqrt(areas / np.pi)
    green = cropped[:, :, 1].astype(float)
    red = cropped[:, :, 0].astype(float)
    green_i, shell_green, red_i, eccentricity, solidity = [], [], [], [], []
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        crop = labels[minr:maxr, minc:maxc] == p.label
        dilated = ndi.binary_dilation(crop, iterations=3)
        shell = dilated & ~crop
        green_i.append(float(green[minr:maxr, minc:maxc][crop].mean()))
        red_i.append(float(red[minr:maxr, minc:maxc][crop].mean()))
        shell_green.append(float(green[minr:maxr, minc:maxc][shell].mean()) if shell.any() else 0.0)
        eccentricity.append(float(p.eccentricity))
        solidity.append(float(p.solidity))

    median_radius = np.median(radii)
    contacts = set()
    nearest_distance = np.zeros(len(props))
    for i in range(len(props)):
        d = np.linalg.norm(centers - centers[i], axis=1)
        d[i] = np.inf
        nearest_distance[i] = d.min()
        close = np.where(d < (radii + radii[i] + 0.75 * median_radius))[0]
        for j in close:
            contacts.add((i, int(j)))
            contacts.add((int(j), i))
    if not contacts:
        for i in range(len(props)):
            d = np.linalg.norm(centers - centers[i], axis=1)
            for j in np.argsort(d)[1:4]:
                contacts.add((i, int(j)))
                contacts.add((int(j), i))

    edge_index = np.array(sorted(contacts), dtype=np.int64).T
    degree = np.bincount(edge_index[0], minlength=len(props)).astype(float)
    total_pixels = labels.shape[0] * labels.shape[1]
    undirected_edges = edge_index.shape[1] / 2
    image_area_um2 = total_pixels * (pixel_size_um ** 2)
    radii_um = radii * pixel_size_um
    nearest_distance_um = nearest_distance * pixel_size_um

    desc = {
        "n_droplets": float(len(props) / image_area_um2 * 1000.0),
        "oil_area_fraction": float(areas.sum() / total_pixels),
        "mean_radius": float(radii_um.mean()),
        "radius_cv": float(radii.std() / (radii.mean() + 1e-8)),
        "mean_degree": float(degree.mean()),
        "max_degree": float(degree.max()),
        "edge_density": float(undirected_edges / max(len(props) * (len(props) - 1) / 2, 1)),
        "mean_nearest_distance": float(nearest_distance_um.mean()),
        "mean_green": float(np.mean(green_i)),
        "mean_shell_green": float(np.mean(shell_green)),
        "mean_eccentricity": float(np.mean(eccentricity)),
        "mean_solidity": float(np.mean(solidity)),
    }
    desc.update(protein_desc)
    return {col: desc.get(col, np.nan) for col in DESC_COLS}


def run_simulation_guided_structure_generator():
    attraction_levels = [
        ("very weak", 0.15),
        ("weak", 0.4),
        ("medium-low", 0.8),
        ("medium-high", 1.2),
        ("strong", 1.8),
        ("very strong", 2.5),
    ]
    n_replicates = 3
    synthetic_rows = []
    image_records = []
    for idx, (label_name, epsilon) in enumerate(attraction_levels):
        for rep in range(n_replicates):
            seed = SEED + 100 * idx + rep
            positions, engine = simulate_droplet_positions_hoomd(epsilon, seed=seed)
            if positions is None:
                positions = simulate_droplet_positions_numpy(epsilon, seed=seed)
                engine = "numpy_surrogate"
            rgb = render_synthetic_confocal_like_image(positions, epsilon)
            desc = descriptors_from_synthetic_rgb(rgb)
            synthetic_rows.append({
                "simulation_id": f"{label_name}_attraction_rep{rep + 1}",
                "attraction_level": label_name,
                "attraction_strength": epsilon,
                "replicate": rep + 1,
                "simulation_engine": engine,
                **desc,
            })
            if rep == 0 and label_name in {"weak", "medium-high", "very strong"}:
                image_records.append((label_name, epsilon, engine, rgb))

    sim_desc = pd.DataFrame(synthetic_rows)
    descriptor_medians = desc_df[DESC_COLS].median(numeric_only=True)
    sim_desc[DESC_COLS] = sim_desc[DESC_COLS].astype(float).fillna(descriptor_medians)
    sim_desc.to_csv(OUTPUT_DIR / "simulation_guided_synthetic_structure_descriptors.csv", index=False)

    descriptor_scaler = StandardScaler().fit(X_desc)
    descriptor_pca = PCA(n_components=2, random_state=SEED).fit(descriptor_scaler.transform(X_desc))
    real_scores = descriptor_pca.transform(descriptor_scaler.transform(X_desc))
    sim_scores = descriptor_pca.transform(descriptor_scaler.transform(sim_desc[DESC_COLS].values.astype(float)))
    sim_pca = sim_desc[["simulation_id", "attraction_level", "attraction_strength", "replicate", "simulation_engine"]].copy()
    sim_pca["PC1"] = sim_scores[:, 0]
    sim_pca["PC2"] = sim_scores[:, 1]
    sim_pca.to_csv(OUTPUT_DIR / "simulation_guided_synthetic_structure_pca_scores.csv", index=False)
    sim_pca_mean = (
        sim_pca
        .groupby(["attraction_level", "attraction_strength"], sort=False)[["PC1", "PC2"]]
        .mean()
        .reset_index()
        .sort_values("attraction_strength")
    )

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8), constrained_layout=True)
    for ax, (label_name, epsilon, engine, rgb) in zip(axes[0], image_records):
        ax.imshow(rgb)
        ax.set_title(f"{label_name} attraction\n{engine}, epsilon={epsilon}", fontsize=13)
        ax.axis("off")

    ax = axes[1, 0]
    real_color = aligned["log_Gp"].values.astype(float)
    sc = ax.scatter(
        real_scores[:, 0],
        real_scores[:, 1],
        c=real_color,
        cmap="viridis",
        s=54,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85,
        label="Real confocal",
    )
    ax.scatter(
        sim_pca["PC1"],
        sim_pca["PC2"],
        marker="x",
        s=42,
        color="#C00000",
        linewidth=1.2,
        alpha=0.55,
        label="Simulated replicates",
    )
    ax.plot(sim_pca_mean["PC1"], sim_pca_mean["PC2"], color="#C00000", linewidth=1.6, alpha=0.8)
    for (_, row), marker in zip(sim_pca_mean.iterrows(), ["o", "s", "^", "D", "P", "X"]):
        ax.scatter(
            row["PC1"],
            row["PC2"],
            marker=marker,
            s=155,
            color="#C00000",
            edgecolor="black",
            linewidth=0.8,
            label=f"Simulated {row['attraction_level']}",
        )
    ax.set_xlabel(f"PC1 ({descriptor_pca.explained_variance_ratio_[0] * 100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({descriptor_pca.explained_variance_ratio_[1] * 100:.1f}% variance)", fontsize=12)
    ax.set_title("Real vs simulated descriptor space", fontsize=13)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Real log$_{10}$ G'$_{1Hz}$ (Pa)", fontsize=11)

    ax = axes[1, 1]
    trend_cols = ["protein_connection_density", "mean_nearest_distance", "mean_shell_green"]
    available_trend_cols = [col for col in trend_cols if col in DESC_COLS]
    trend_summary = (
        sim_desc
        .groupby("attraction_strength")[available_trend_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    trend_summary.to_csv(OUTPUT_DIR / "simulation_guided_shap_descriptor_trends.csv", index=False)

    colors = {
        "protein_connection_density": "#C00000",
        "mean_nearest_distance": "#4C78A8",
        "mean_shell_green": "#54A24B",
    }
    labels = {
        "protein_connection_density": "Protein connectivity",
        "mean_nearest_distance": "Nearest-neighbor distance",
        "mean_shell_green": "Interfacial protein intensity",
    }
    for col in available_trend_cols:
        means = trend_summary[(col, "mean")].values.astype(float)
        stds = trend_summary[(col, "std")].fillna(0).values.astype(float)
        xvals = trend_summary[("attraction_strength", "")].values.astype(float)
        if np.nanmax(means) > np.nanmin(means):
            norm_means = (means - np.nanmin(means)) / (np.nanmax(means) - np.nanmin(means))
            norm_stds = stds / (np.nanmax(means) - np.nanmin(means))
        else:
            norm_means = means
            norm_stds = stds
        ax.errorbar(
            xvals,
            norm_means,
            yerr=norm_stds,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors.get(col, "0.25"),
            label=labels.get(col, col),
        )
    ax.set_xlabel("Simulation attraction strength", fontsize=12)
    ax.set_ylabel("Normalized descriptor value", fontsize=12)
    ax.set_title("SHAP/PCA descriptors vary with attraction", fontsize=13)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1, 2]
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "Interpretation claim:\n"
        "Simulated weak-to-strong attraction\n"
        "is compared in the same descriptor\n"
        "space as real confocal images.\n\n"
        "Use this as mechanism support:\n"
        "SHAP/PCA descriptor trends change\n"
        "with attraction and can be compared\n"
        "with the experimental high-G'\n"
        "direction.\n\n"
        "Do not claim direct MPCD prediction\n"
        "of absolute G'.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="0.82", boxstyle="round,pad=0.45"),
    )

    for ax in axes[1, :2]:
        ax.grid(color="0.9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=10)

    save_fig("figure_05e_simulation_guided_structure_generator")
    plt.show()


run_simulation_guided_structure_generator()

INPUT_SETS = {
    "Formulation only": X_form_aligned,
    "Structure only": X_desc,
    "Formulation + structure": X_desc_form,
}


def make_rbf_gpr():
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1e1))
    )
    return Pipeline([
        ("scale", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=20,
            random_state=SEED,
        )),
    ])


def make_pls(n_features: int):
    min_train_size = min(len(train_idx) for train_idx, _ in splits)
    n_components = max(1, min(3, n_features, min_train_size - 1))
    return Pipeline([
        ("scale", StandardScaler()),
        ("pls", PLSRegression(n_components=n_components)),
    ])


def make_model_library(n_features: int) -> dict:
    models = {
        "Ridge": Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),
        "PLS Regression": make_pls(n_features),
        "RBF-GPR": make_rbf_gpr(),
        "SVR": Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(C=3.0, epsilon=0.08, kernel="rbf")),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            random_state=SEED,
            min_samples_leaf=2,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            random_state=SEED,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            random_state=SEED,
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=1,
        )
    else:
        print("XGBoost is not installed; skipping XGBoost. Install with `%pip install xgboost`.")
    return models


def cv_sklearn(X, y_values, splits_in, estimator, model_name, input_set_name, target_config: dict = GP_TARGET):
    pred = np.zeros_like(y_values, dtype=float)
    for train_idx, test_idx in splits_in:
        model = copy.deepcopy(estimator)
        model.fit(X[train_idx], y_values[train_idx])
        fold_pred = model.predict(X[test_idx])
        pred[test_idx] = np.asarray(fold_pred).reshape(-1)

    rmse = float(np.sqrt(np.mean((y_values - pred) ** 2)))
    mae = mean_absolute_error(y_values, pred)
    r2 = r2_score(y_values, pred)
    out = pd.DataFrame({
        "algorithm": model_name,
        "input_set": input_set_name,
        "model": f"{model_name} | {input_set_name}",
        "target": target_config["key"],
        "sample_id": aligned["image_file"],
        target_config["true_log_col"]: y_values,
        target_config["pred_log_col"]: pred,
        target_config["true_value_col"]: 10 ** y_values,
        target_config["pred_value_col"]: 10 ** pred,
    })
    print(
        f"{target_config['value_label']:16s} | {model_name:18s} | {input_set_name:25s} "
        f"MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.3f}"
    )
    return out


def summarize_predictions(results: pd.DataFrame, target_config: dict, group_cols: list[str]) -> pd.DataFrame:
    true_log_col = target_config["true_log_col"]
    pred_log_col = target_config["pred_log_col"]
    return (
        results.groupby(group_cols)
        .apply(lambda g: pd.Series({
            "n_test": len(g),
            "MAE_log10": mean_absolute_error(g[true_log_col], g[pred_log_col]),
            "RMSE_log10": float(np.sqrt(np.mean((g[true_log_col] - g[pred_log_col]) ** 2))),
            "R2_log10": r2_score(g[true_log_col], g[pred_log_col]) if len(g) > 1 else np.nan,
        }))
        .reset_index()
    )


def run_primary_benchmark(target_config: dict):
    y_values = aligned[target_config["log_col"]].values.astype(float)
    benchmark_results = []
    for input_set_name, X in INPUT_SETS.items():
        for model_name, estimator in make_model_library(X.shape[1]).items():
            benchmark_results.append(cv_sklearn(X, y_values, splits, estimator, model_name, input_set_name, target_config))

    results = pd.concat(benchmark_results, ignore_index=True)
    summary_out = (
        summarize_predictions(results, target_config, ["algorithm", "input_set"])
        .sort_values(["RMSE_log10", "MAE_log10"])
    )

    display(summary_out)
    output_tag = target_config["output_tag"]
    results.to_csv(OUTPUT_DIR / f"{output_tag}model_comparison_grouped_cv_predictions.csv", index=False)
    summary_out.to_csv(OUTPUT_DIR / f"{output_tag}model_comparison_grouped_cv_summary.csv", index=False)
    return results, summary_out


benchmark_outputs = {}
for target_config in TARGET_CONFIGS:
    benchmark_outputs[target_config["key"]] = run_primary_benchmark(target_config)

all_results, summary = benchmark_outputs["Gp"]

# Direct comparison to the previous non-stratified grouped CV split.
def run_split_strategy_comparison_data(target_config: dict):
    y_values = aligned[target_config["log_col"]].values.astype(float)
    split_strategy_results = []
    for split_strategy_name, split_list in [
        ("Protein-stratified grouped CV", splits),
        ("Non-stratified grouped CV", grouped_splits),
    ]:
        for input_set_name, X in INPUT_SETS.items():
            for model_name, estimator in make_model_library(X.shape[1]).items():
                out = cv_sklearn(X, y_values, split_list, estimator, model_name, input_set_name, target_config)
                out["split_strategy"] = split_strategy_name
                split_strategy_results.append(out)

    predictions = pd.concat(split_strategy_results, ignore_index=True)
    summary_out = (
        summarize_predictions(predictions, target_config, ["split_strategy", "algorithm", "input_set"])
        .sort_values(["input_set", "algorithm", "split_strategy"])
    )
    display(summary_out)
    output_tag = target_config["output_tag"]
    predictions.to_csv(OUTPUT_DIR / f"{output_tag}stratified_vs_non_stratified_grouped_cv_predictions.csv", index=False)
    summary_out.to_csv(OUTPUT_DIR / f"{output_tag}stratified_vs_non_stratified_grouped_cv_summary.csv", index=False)
    return predictions, summary_out


split_strategy_outputs = {}
for target_config in TARGET_CONFIGS:
    split_strategy_outputs[target_config["key"]] = run_split_strategy_comparison_data(target_config)

split_strategy_predictions, split_strategy_summary = split_strategy_outputs["Gp"]

# %%
# 9b. Protein-type validation modes
#
# These extra analyses answer different generalization questions:
#   1. Both proteins -> both proteins, stratified grouped CV
#   2. Train FPI -> test QPI
#   3. Train QPI -> test FPI
#   4. FPI-only grouped CV
#   5. QPI-only grouped CV
#
# All grouped CV modes still use formulation groups to avoid replicate-image
# leakage.

protein_types = aligned["protein type"].astype(str).values
all_indices = np.arange(len(aligned))


def _prediction_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_indices: np.ndarray,
    algorithm: str,
    input_set: str,
    validation_mode: str,
    train_protein: str,
    test_protein: str,
    target_config: dict,
) -> pd.DataFrame:
    return pd.DataFrame({
        "validation_mode": validation_mode,
        "algorithm": algorithm,
        "input_set": input_set,
        "model": f"{algorithm} | {input_set}",
        "target": target_config["key"],
        "sample_id": aligned.iloc[row_indices]["image_file"].values,
        "protein_type": aligned.iloc[row_indices]["protein type"].values,
        "formulation_group": aligned.iloc[row_indices]["formulation_group"].values,
        "train_protein": train_protein,
        "test_protein": test_protein,
        target_config["true_log_col"]: y_true,
        target_config["pred_log_col"]: y_pred,
        target_config["true_value_col"]: 10 ** y_true,
        target_config["pred_value_col"]: 10 ** y_pred,
    })


def evaluate_fixed_train_test(
    X: np.ndarray,
    y_values: np.ndarray,
    estimator,
    algorithm: str,
    input_set: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    validation_mode: str,
    train_protein: str,
    test_protein: str,
    target_config: dict,
) -> pd.DataFrame:
    model = copy.deepcopy(estimator)
    model.fit(X[train_idx], y_values[train_idx])
    pred = np.asarray(model.predict(X[test_idx])).reshape(-1)
    return _prediction_frame(
        y_values[test_idx],
        pred,
        test_idx,
        algorithm,
        input_set,
        validation_mode,
        train_protein,
        test_protein,
        target_config,
    )


def evaluate_grouped_cv_subset(
    X: np.ndarray,
    y_values: np.ndarray,
    estimator,
    algorithm: str,
    input_set: str,
    subset_idx: np.ndarray,
    validation_mode: str,
    protein_label: str,
    target_config: dict,
) -> pd.DataFrame:
    subset_groups = groups[subset_idx]
    n_subset_splits = min(5, len(np.unique(subset_groups)))
    subset_splits = list(GroupKFold(n_splits=n_subset_splits).split(subset_idx, groups=subset_groups))
    pred = np.zeros(len(subset_idx), dtype=float)
    for train_local, test_local in subset_splits:
        train_idx = subset_idx[train_local]
        test_idx = subset_idx[test_local]
        model = copy.deepcopy(estimator)
        model.fit(X[train_idx], y_values[train_idx])
        pred[test_local] = np.asarray(model.predict(X[test_idx])).reshape(-1)

    return _prediction_frame(
        y_values[subset_idx],
        pred,
        subset_idx,
        algorithm,
        input_set,
        validation_mode,
        protein_label,
        protein_label,
        target_config,
    )


def run_protein_validation(target_config: dict, primary_results: pd.DataFrame):
    y_values = aligned[target_config["log_col"]].values.astype(float)
    protein_validation_results = []

    # Reuse the primary both-protein stratified grouped CV predictions.
    sample_to_protein = aligned.set_index("image_file")["protein type"].astype(str).to_dict()
    sample_to_group = aligned.set_index("image_file")["formulation_group"].astype(str).to_dict()
    both_protein_results = primary_results.copy()
    both_protein_results["validation_mode"] = "Both proteins stratified grouped CV"
    both_protein_results["protein_type"] = both_protein_results["sample_id"].map(sample_to_protein)
    both_protein_results["formulation_group"] = both_protein_results["sample_id"].map(sample_to_group)
    both_protein_results["train_protein"] = "FPI+QPI"
    both_protein_results["test_protein"] = "FPI+QPI"
    protein_validation_results.append(both_protein_results)

    for input_set_name, X in INPUT_SETS.items():
        model_library = make_model_library(X.shape[1])
        for model_name, estimator in model_library.items():
            fpi_idx = all_indices[protein_types == "FPI"]
            qpi_idx = all_indices[protein_types == "QPI"]

            if len(fpi_idx) and len(qpi_idx):
                protein_validation_results.append(
                    evaluate_fixed_train_test(
                        X, y_values, estimator, model_name, input_set_name,
                        train_idx=fpi_idx,
                        test_idx=qpi_idx,
                        validation_mode="Train FPI, test QPI",
                        train_protein="FPI",
                        test_protein="QPI",
                        target_config=target_config,
                    )
                )
                protein_validation_results.append(
                    evaluate_fixed_train_test(
                        X, y_values, estimator, model_name, input_set_name,
                        train_idx=qpi_idx,
                        test_idx=fpi_idx,
                        validation_mode="Train QPI, test FPI",
                        train_protein="QPI",
                        test_protein="FPI",
                        target_config=target_config,
                    )
                )

            for protein_label, subset_idx in [("FPI", fpi_idx), ("QPI", qpi_idx)]:
                if len(np.unique(groups[subset_idx])) >= 2:
                    protein_validation_results.append(
                        evaluate_grouped_cv_subset(
                            X, y_values, estimator, model_name, input_set_name,
                            subset_idx=subset_idx,
                            validation_mode=f"{protein_label}-only grouped CV",
                            protein_label=protein_label,
                            target_config=target_config,
                        )
                    )

    predictions = pd.concat(protein_validation_results, ignore_index=True)
    summary_out = (
        summarize_predictions(
            predictions,
            target_config,
            ["validation_mode", "algorithm", "input_set", "train_protein", "test_protein"],
        )
        .sort_values(["validation_mode", "RMSE_log10", "MAE_log10"])
    )

    display(summary_out)
    output_tag = target_config["output_tag"]
    predictions.to_csv(OUTPUT_DIR / f"{output_tag}protein_validation_mode_predictions.csv", index=False)
    summary_out.to_csv(OUTPUT_DIR / f"{output_tag}protein_validation_mode_summary.csv", index=False)

    # Save each validation mode separately so the FPI/QPI analyses are easy to find.
    safe_mode_names = {
        "Both proteins stratified grouped CV": "both_proteins_stratified_grouped_cv",
        "Train FPI, test QPI": "train_FPI_test_QPI",
        "Train QPI, test FPI": "train_QPI_test_FPI",
        "FPI-only grouped CV": "FPI_only_grouped_cv",
        "QPI-only grouped CV": "QPI_only_grouped_cv",
    }
    for mode_name, safe_name in safe_mode_names.items():
        mode_summary = summary_out[summary_out["validation_mode"] == mode_name]
        mode_predictions = predictions[predictions["validation_mode"] == mode_name]
        mode_summary.to_csv(OUTPUT_DIR / f"{output_tag}protein_validation_summary_{safe_name}.csv", index=False)
        mode_predictions.to_csv(OUTPUT_DIR / f"{output_tag}protein_validation_predictions_{safe_name}.csv", index=False)

    print(f"\nProtein-type validation modes saved for {target_config['value_label']}:")
    for mode_name, safe_name in safe_mode_names.items():
        n_rows = len(summary_out[summary_out["validation_mode"] == mode_name])
        print(f"  {mode_name}: {n_rows} model/input rows -> {output_tag}protein_validation_summary_{safe_name}.csv")

    best = (
        summary_out
        .sort_values(["validation_mode", "RMSE_log10", "MAE_log10"])
        .groupby("validation_mode")
        .head(5)
        .reset_index(drop=True)
    )
    display(best)
    best.to_csv(OUTPUT_DIR / f"{output_tag}protein_validation_best_5_per_mode.csv", index=False)
    return predictions, summary_out, best


protein_validation_outputs = {}
for target_config in TARGET_CONFIGS:
    primary_results, _ = benchmark_outputs[target_config["key"]]
    protein_validation_outputs[target_config["key"]] = run_protein_validation(target_config, primary_results)

protein_validation_predictions, protein_validation_summary, protein_validation_best = protein_validation_outputs["Gp"]


def plot_protein_validation_mode_bars(metric_name: str, ylabel: str, filename: str, input_set_filter: str = "Formulation + structure"):
    """Compare FPI/QPI validation modes for each model using one input set."""
    plot_df = protein_validation_summary[protein_validation_summary["input_set"] == input_set_filter].copy()
    mode_order = [
        "Both proteins stratified grouped CV",
        "FPI-only grouped CV",
        "QPI-only grouped CV",
        "Train FPI, test QPI",
        "Train QPI, test FPI",
    ]
    mode_order = [m for m in mode_order if m in plot_df["validation_mode"].unique()]
    algo_order = [
        "RBF-GPR",
        "PLS Regression",
        "XGBoost",
        "Gradient Boosting",
        "Extra Trees",
        "Random Forest",
        "SVR",
        "Ridge",
    ]
    algo_order = [a for a in algo_order if a in plot_df["algorithm"].unique()]

    mat = (
        plot_df
        .pivot(index="algorithm", columns="validation_mode", values=metric_name)
        .reindex(index=algo_order, columns=mode_order)
    )

    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    x = np.arange(len(algo_order))
    width = min(0.16, 0.75 / max(len(mode_order), 1))
    colors = ["#4C78A8", "#54A24B", "#72B7B2", "#F58518", "#E45756"]
    offsets = (np.arange(len(mode_order)) - (len(mode_order) - 1) / 2) * width
    for offset, mode_name, color in zip(offsets, mode_order, colors):
        ax.bar(
            x + offset,
            mat[mode_name].values.astype(float),
            width=width,
            label=mode_name,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algo_order, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(input_set_filter, fontsize=14)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    if metric_name == "R2_log10":
        ax.axhline(0, color="0.4", linewidth=0.8)
    else:
        ax.set_ylim(0, np.nanmax(mat.values.astype(float)) * 1.18)
    plt.tight_layout()
    save_fig(filename)
    plt.show()


# Protein-validation mode CSVs are saved above, but only the main
# both-proteins stratified grouped CV figures are generated for publication.


def plot_split_strategy_comparison(metric_name: str, ylabel: str, filename: str, input_set_filter: str = "Formulation + structure"):
    """Compare protein-stratified grouped CV with previous non-stratified grouped CV."""
    plot_df = split_strategy_summary[split_strategy_summary["input_set"] == input_set_filter].copy()
    strategy_order = ["Protein-stratified grouped CV", "Non-stratified grouped CV"]
    strategy_order = [s for s in strategy_order if s in plot_df["split_strategy"].unique()]
    algo_order = [
        "RBF-GPR",
        "PLS Regression",
        "XGBoost",
        "Gradient Boosting",
        "Extra Trees",
        "Random Forest",
        "SVR",
        "Ridge",
    ]
    algo_order = [a for a in algo_order if a in plot_df["algorithm"].unique()]

    mat = (
        plot_df
        .pivot(index="algorithm", columns="split_strategy", values=metric_name)
        .reindex(index=algo_order, columns=strategy_order)
    )

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x = np.arange(len(algo_order))
    width = 0.34
    colors = {
        "Protein-stratified grouped CV": "#4C78A8",
        "Non-stratified grouped CV": "#F58518",
    }
    offsets = [-width / 2, width / 2]
    for offset, strategy_name in zip(offsets, strategy_order):
        ax.bar(
            x + offset,
            mat[strategy_name].values.astype(float),
            width=width,
            label=strategy_name,
            color=colors[strategy_name],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algo_order, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(input_set_filter, fontsize=14)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    if metric_name == "R2_log10":
        ax.axhline(0, color="0.4", linewidth=0.8)
    else:
        ax.set_ylim(0, np.nanmax(mat.values.astype(float)) * 1.18)
    plt.tight_layout()
    save_fig(filename)
    plt.show()


# The stratified vs non-stratified comparison is saved as CSV above. Figures are
# intentionally not generated here to keep the publication output focused.

# %%
# 10. Publication-level model figures

INPUT_ORDER = ["Formulation only", "Structure only", "Formulation + structure"]
ALGORITHM_ORDER = [
    "RBF-GPR",
    "PLS Regression",
    "XGBoost",
    "Gradient Boosting",
    "Extra Trees",
    "Random Forest",
    "SVR",
    "Ridge",
]


def metric_matrix(summary_in: pd.DataFrame, algorithm_order: list[str], metric_name: str) -> pd.DataFrame:
    return (
        summary_in.pivot(index="algorithm", columns="input_set", values=metric_name)
        .reindex(index=algorithm_order, columns=INPUT_ORDER)
    )


def plot_metric_heatmap(summary_in: pd.DataFrame, algorithm_order: list[str], target_config: dict, filename: str):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), facecolor="white")
    for ax, metric, label, cmap in [
        (axes[0], "R2_log10", r"$R^2$", "YlGnBu"),
        (axes[1], "RMSE_log10", "RMSE", "YlOrRd_r"),
        (axes[2], "MAE_log10", "MAE", "YlOrRd_r"),
    ]:
        mat = metric_matrix(summary_in, algorithm_order, metric)
        cmap_obj = copy.copy(plt.get_cmap(cmap))
        cmap_obj.set_bad(color="white")
        values = np.ma.masked_invalid(mat.values.astype(float))
        im = ax.imshow(values, cmap=cmap_obj, aspect="auto")
        ax.set_facecolor("white")
        ax.set_xticks(np.arange(len(INPUT_ORDER)))
        ax.set_xticklabels(INPUT_ORDER, rotation=25, ha="right", fontsize=13)
        ax.set_yticks(np.arange(len(algorithm_order)))
        ax.set_yticklabels(algorithm_order, fontsize=13)
        ax.set_title(label, fontsize=16, pad=10)
        ax.set_xticks(np.arange(-0.5, len(INPUT_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(algorithm_order), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
        ax.tick_params(axis="both", length=0)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                if pd.notna(val):
                    rgba = im.cmap(im.norm(val))
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    text_color = "white" if luminance < 0.48 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color=text_color,
                        fontweight="bold",
                    )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    save_fig(filename)
    plt.show()


def plot_grouped_metric_bars(
    summary_in: pd.DataFrame,
    algorithm_order: list[str],
    metric_name: str,
    ylabel: str,
    filename: str,
    higher_is_better: bool = True,
):
    metric_df = metric_matrix(summary_in, algorithm_order, metric_name)
    x = np.arange(len(algorithm_order))
    width = 0.24
    colors = {
        "Formulation only": "#4C78A8",
        "Structure only": "#F58518",
        "Formulation + structure": "#54A24B",
    }

    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    for offset, input_set in zip([-width, 0, width], INPUT_ORDER):
        values = metric_df[input_set].values.astype(float)
        ax.bar(
            x + offset,
            values,
            width=width,
            label=input_set,
            color=colors[input_set],
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(algorithm_order, rotation=30, ha="right", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.8)
    ax.set_axisbelow(True)
    if higher_is_better:
        ax.set_ylim(min(0, np.nanmin(metric_df.values) - 0.08), min(1.05, np.nanmax(metric_df.values) + 0.12))
    else:
        ax.set_ylim(0, np.nanmax(metric_df.values) * 1.18)

    plt.tight_layout()
    save_fig(filename)
    plt.show()


def algorithm_chunks(algorithm_order: list[str], chunk_size: int = 3):
    for start in range(0, len(algorithm_order), chunk_size):
        yield start // chunk_size + 1, algorithm_order[start:start + chunk_size]


def fitted_parity_line(x: pd.Series, y_values: pd.Series):
    """Fit log10(predicted) = a * log10(measured) + b for log-log parity axes."""
    x_log = np.log10(x.astype(float).to_numpy())
    y_log = np.log10(y_values.astype(float).to_numpy())
    finite = np.isfinite(x_log) & np.isfinite(y_log)
    x_log = x_log[finite]
    y_log = y_log[finite]
    if len(x_log) < 2 or np.nanstd(x_log) == 0:
        return None
    slope, intercept = np.polyfit(x_log, y_log, 1)
    fit_log = slope * x_log + intercept
    ss_res = np.sum((y_log - fit_log) ** 2)
    ss_tot = np.sum((y_log - y_log.mean()) ** 2)
    fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(intercept), float(fit_r2)


def plot_grid_parity(results: pd.DataFrame, summary_in: pd.DataFrame, algorithm_order: list[str], target_config: dict, filename_prefix: str):
    true_value_col = target_config["true_value_col"]
    pred_value_col = target_config["pred_value_col"]
    lo = min(results[true_value_col].min(), results[pred_value_col].min())
    hi = max(results[true_value_col].max(), results[pred_value_col].max())
    fit_x = np.logspace(np.log10(lo), np.log10(hi), 200)

    for chunk_id, algorithms in algorithm_chunks(algorithm_order, chunk_size=4):
        nrows = 4
        ncols = len(INPUT_ORDER)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 15.0), squeeze=False)

        for i, algorithm in enumerate(algorithms):
            for j, input_set in enumerate(INPUT_ORDER):
                ax = axes[i, j]
                sub = results[(results["algorithm"] == algorithm) & (results["input_set"] == input_set)]
                if sub.empty:
                    ax.axis("off")
                    continue
                metric_row = summary_in[
                    (summary_in["algorithm"] == algorithm) & (summary_in["input_set"] == input_set)
                ].iloc[0]
                ax.scatter(
                    sub[true_value_col],
                    sub[pred_value_col],
                    s=58,
                    alpha=0.86,
                    color="#2F5597",
                    edgecolor="white",
                    linewidth=0.7,
                )
                ax.plot([lo, hi], [lo, hi], color="0.25", linestyle="--", linewidth=1.3, label="y=x")
                fit = fitted_parity_line(sub[true_value_col], sub[pred_value_col])
                if fit is not None:
                    slope, intercept, fit_r2 = fit
                    fit_y = 10 ** (slope * np.log10(fit_x) + intercept)
                    ax.plot(fit_x, fit_y, color="#C00000", linestyle="-", linewidth=1.8)
                    ax.text(
                        0.04,
                        0.96,
                        fr"$\log_{{10}}y={slope:.2f}\log_{{10}}x{intercept:+.2f}$"
                        + "\n"
                        + fr"$R^2$={fit_r2:.2f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=10.5,
                        color="#C00000",
                        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="0.85", alpha=0.90),
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.tick_params(axis="both", labelsize=10)
                ax.set_title(
                    f"{algorithm}\n{input_set}\nR2={metric_row['R2_log10']:.2f}, RMSE={metric_row['RMSE_log10']:.2f}",
                    fontsize=12.5,
                    fontweight="semibold",
                    pad=10,
                )
                ax.set_xlabel(target_config["measured_label"], fontsize=12, fontweight="semibold", labelpad=6)
                ax.set_ylabel(target_config["predicted_label"], fontsize=12, fontweight="semibold", labelpad=6)

        for empty_i in range(len(algorithms), nrows):
            for j in range(ncols):
                axes[empty_i, j].axis("off")

        plt.tight_layout()
        save_fig(f"{filename_prefix}_{chunk_id}")
        plt.show()


def plot_grid_bland_altman(results: pd.DataFrame, algorithm_order: list[str], target_config: dict, filename_prefix: str):
    true_log_col = target_config["true_log_col"]
    pred_log_col = target_config["pred_log_col"]
    is_breaking_stress = target_config["key"] == "breaking_stress"
    ba_x_label = (
        r"Mean log$_{10}$ breaking stress (Pa)"
        if is_breaking_stress
        else r"Mean log$_{10}$ G'$_{1Hz}$ (Pa)"
    )
    ba_y_label = (
        r"$\Delta$ log$_{10}$ breaking stress (Pa)"
        if is_breaking_stress
        else r"$\Delta$ log$_{10}$ G'$_{1Hz}$ (Pa)"
    )
    for chunk_id, algorithms in algorithm_chunks(algorithm_order, chunk_size=4):
        nrows = 4
        ncols = len(INPUT_ORDER)
        ba_figsize = (16.0, 15.8)
        fig, axes = plt.subplots(nrows, ncols, figsize=ba_figsize, squeeze=False)

        for i, algorithm in enumerate(algorithms):
            algorithm_sub = results[results["algorithm"] == algorithm]
            algorithm_diff = algorithm_sub[pred_log_col].values - algorithm_sub[true_log_col].values
            if len(algorithm_diff) and np.isfinite(algorithm_diff).any():
                y_abs = np.nanmax(np.abs(algorithm_diff))
                y_abs = max(y_abs * 1.18, 0.05)
                row_ylim = (-y_abs, y_abs)
            else:
                row_ylim = None

            for j, input_set in enumerate(INPUT_ORDER):
                ax = axes[i, j]
                sub = results[(results["algorithm"] == algorithm) & (results["input_set"] == input_set)]
                if sub.empty:
                    ax.axis("off")
                    continue
                measured = sub[true_log_col].values
                predicted = sub[pred_log_col].values
                mean_log = 0.5 * (measured + predicted)
                diff = predicted - measured
                bias = diff.mean()
                sd = diff.std(ddof=1)
                ax.scatter(
                    mean_log,
                    diff,
                    s=58,
                    alpha=0.86,
                    color="#8C564B",
                    edgecolor="white",
                    linewidth=0.7,
                )
                ax.axhline(0, color="0.55", linestyle=":", linewidth=1.2)
                ax.axhline(bias, color="black", linewidth=1.3)
                ax.axhline(bias + 1.96 * sd, color="tab:red", linestyle="--", linewidth=1.2)
                ax.axhline(bias - 1.96 * sd, color="tab:red", linestyle="--", linewidth=1.2)
                if row_ylim is not None:
                    ax.set_ylim(row_ylim)
                ax.tick_params(axis="both", labelsize=10)
                ax.set_title(
                    f"{algorithm}\n{input_set}\nbias={bias:.2f}",
                    fontsize=12.5,
                    fontweight="semibold",
                    pad=10,
                )
                ax.set_xlabel(ba_x_label, fontsize=11, fontweight="semibold", labelpad=7)
                ax.set_ylabel(ba_y_label, fontsize=11, fontweight="semibold", labelpad=8)

        for empty_i in range(len(algorithms), nrows):
            for j in range(ncols):
                axes[empty_i, j].axis("off")

        wspace = 0.58
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.06, top=0.95, wspace=wspace, hspace=0.58)
        save_fig(f"{filename_prefix}_{chunk_id}")
        plt.show()


def make_publication_model_figures(target_config: dict, results: pd.DataFrame, summary_in: pd.DataFrame):
    algorithm_order = [m for m in ALGORITHM_ORDER if m in summary_in["algorithm"].unique()]
    offset = target_config["figure_offset"]
    prefix = target_config["output_tag"]

    metrics_fig = "figure_06_model_comparison_metrics" if offset == 0 else f"figure_{6 + offset:02d}_breaking_stress_model_comparison_metrics"
    r2_fig = "figure_06b_model_comparison_grouped_bar_R2" if offset == 0 else f"figure_{6 + offset:02d}b_breaking_stress_model_comparison_grouped_bar_R2"
    rmse_fig = "figure_06c_model_comparison_grouped_bar_RMSE" if offset == 0 else f"figure_{6 + offset:02d}c_breaking_stress_model_comparison_grouped_bar_RMSE"
    parity_prefix = "figure_07_parity_4x3_algorithm_input_part" if offset == 0 else f"figure_{7 + offset:02d}_breaking_stress_parity_4x3_algorithm_input_part"
    bland_prefix = "figure_08_bland_altman_4x3_algorithm_input_part" if offset == 0 else f"figure_{8 + offset:02d}_breaking_stress_bland_altman_4x3_algorithm_input_part"

    plot_metric_heatmap(summary_in, algorithm_order, target_config, metrics_fig)
    plot_grouped_metric_bars(summary_in, algorithm_order, "R2_log10", target_config["bar_r2_label"], r2_fig, higher_is_better=True)
    plot_grouped_metric_bars(summary_in, algorithm_order, "RMSE_log10", target_config["bar_rmse_label"], rmse_fig, higher_is_better=False)
    plot_grid_parity(results, summary_in, algorithm_order, target_config, parity_prefix)
    plot_grid_bland_altman(results, algorithm_order, target_config, bland_prefix)


for target_config in TARGET_CONFIGS:
    target_results, target_summary = benchmark_outputs[target_config["key"]]
    make_publication_model_figures(target_config, target_results, target_summary)

# %%
# 11. SHAP analysis for formulation GPR, excluding protein type

def clean_feature_names(names):
    return [
        str(name)
        .replace("cat__", "")
        .replace("num__", "")
        .replace("heat_", "Heat: ")
        .replace("protein conc_percentage", "Protein concentration")
        .replace("oil vol_percentage", "Oil volume fraction")
        .replace("NaCl_mM", "NaCl")
        .replace("CaCl2_mM", "CaCl2")
        for name in names
    ]


FORMULATION_SHAP_CAT_COLS = ["heat"]
FORMULATION_SHAP_NUM_COLS = NUM_COLS
formulation_shap_transformer = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), FORMULATION_SHAP_CAT_COLS),
        ("num", StandardScaler(), FORMULATION_SHAP_NUM_COLS),
    ]
)
X_form_shap = formulation_shap_transformer.fit_transform(aligned).astype(float)
FORMULATION_SHAP_NAMES = clean_feature_names(formulation_shap_transformer.get_feature_names_out())


def run_gpr_formulation_shap(target_config: dict = GP_TARGET):
    """
    Explain formulation-only GPR after excluding protein type.

    This keeps the formulation interpretation focused on heat, protein
    concentration, oil fraction, pH, salt, and calcium level.
    """
    try:
        import shap
    except ImportError:
        print("SHAP analysis skipped: install shap with `%pip install shap` and rerun this cell.")
        return None

    model = make_rbf_gpr()
    y_values = aligned[target_config["log_col"]].values.astype(float)
    model.fit(X_form_shap, y_values)

    background_n = min(20, len(X_form_shap))
    explain_n = min(50, len(X_form_shap))
    form_frame = pd.DataFrame(X_form_shap, columns=FORMULATION_SHAP_NAMES)
    background = shap.sample(form_frame, background_n, random_state=SEED)
    explain = form_frame.sample(explain_n, random_state=SEED)

    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = np.asarray(explainer.shap_values(explain, nsamples=200))

    shap_df = pd.DataFrame(shap_values, columns=FORMULATION_SHAP_NAMES, index=explain.index)
    shap_long = (
        shap_df.reset_index(names="row_index")
        .melt(id_vars="row_index", var_name="feature", value_name="shap_value")
    )
    shap_long["sample_id"] = shap_long["row_index"].map(aligned["image_file"].to_dict())
    output_tag = target_config["output_tag"]
    shap_long.to_csv(OUTPUT_DIR / f"{output_tag}formulation_gpr_shap_values_without_protein_type.csv", index=False)

    shap_summary = (
        shap_df.abs().mean()
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("mean_abs_shap", ascending=False)
    )
    display(shap_summary)
    shap_summary.to_csv(OUTPUT_DIR / f"{output_tag}formulation_gpr_shap_summary_without_protein_type.csv", index=False)

    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, explain, feature_names=FORMULATION_SHAP_NAMES, show=False, plot_size=None)
    plt.tight_layout()
    summary_fig = (
        "figure_09_formulation_shap_without_protein_type_summary"
        if target_config["key"] == "Gp"
        else "figure_29_breaking_stress_formulation_shap_without_protein_type_summary"
    )
    save_fig(summary_fig)
    plt.show()

    top = shap_summary.head(10)
    plt.figure(figsize=(7, 4.5))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#4C78A8")
    plt.gca().invert_yaxis()
    plt.xlabel(target_config["shap_xlabel"])
    plt.title(target_config["formulation_shap_title"])
    plt.tight_layout()
    bar_fig = (
        "figure_10_formulation_shap_without_protein_type_bar"
        if target_config["key"] == "Gp"
        else "figure_30_breaking_stress_formulation_shap_without_protein_type_bar"
    )
    save_fig(bar_fig)
    plt.show()

    return {
        "summary": shap_summary,
        "long": shap_long,
        "shap_values": shap_values,
        "explain": explain,
        "feature_names": FORMULATION_SHAP_NAMES,
    }


gpr_formulation_shap_outputs = {
    target_config["key"]: run_gpr_formulation_shap(target_config)
    for target_config in TARGET_CONFIGS
}
gpr_formulation_shap = gpr_formulation_shap_outputs["Gp"]

# %%
# 12. SHAP analysis for microstructural descriptor GPR

def run_gpr_descriptor_shap(target_config: dict = GP_TARGET):
    """Explain structure-only GPR using publication-friendly descriptor names."""
    try:
        import shap
    except ImportError:
        print("Descriptor SHAP analysis skipped: install shap with `%pip install shap` and rerun this cell.")
        return None

    model = make_rbf_gpr()
    y_values = aligned[target_config["log_col"]].values.astype(float)
    model.fit(X_desc, y_values)

    desc_display_names = [DESCRIPTOR_PUBLICATION_LABELS[c] for c in DESC_COLS]
    desc_frame = pd.DataFrame(X_desc, columns=desc_display_names)
    background_n = min(20, len(desc_frame))
    explain_n = min(50, len(desc_frame))
    background = shap.sample(desc_frame, background_n, random_state=SEED)
    explain = desc_frame.sample(explain_n, random_state=SEED)

    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = np.asarray(explainer.shap_values(explain, nsamples=200))

    shap_df = pd.DataFrame(shap_values, columns=desc_display_names, index=explain.index)
    shap_long = (
        shap_df.reset_index(names="row_index")
        .melt(id_vars="row_index", var_name="feature", value_name="shap_value")
    )
    shap_long["sample_id"] = shap_long["row_index"].map(aligned["image_file"].to_dict())
    output_tag = target_config["output_tag"]
    shap_long.to_csv(OUTPUT_DIR / f"{output_tag}microstructural_descriptor_gpr_shap_values_publication_names.csv", index=False)

    shap_summary = (
        shap_df.abs().mean()
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "publication_feature"})
        .sort_values("mean_abs_shap", ascending=False)
    )
    display(shap_summary)
    shap_summary.to_csv(OUTPUT_DIR / f"{output_tag}microstructural_descriptor_gpr_shap_summary_publication_names.csv", index=False)

    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, explain, feature_names=desc_display_names, show=False, plot_size=None)
    plt.tight_layout()
    summary_fig = (
        "figure_11_microstructural_descriptor_shap_summary_publication_names"
        if target_config["key"] == "Gp"
        else "figure_31_breaking_stress_microstructural_descriptor_shap_summary_publication_names"
    )
    save_fig(summary_fig)
    plt.show()

    top = shap_summary.head(10)
    plt.figure(figsize=(7, 4.8))
    plt.barh(top["publication_feature"], top["mean_abs_shap"], color="#54A24B")
    plt.gca().invert_yaxis()
    plt.xlabel(target_config["shap_xlabel"])
    plt.title(target_config["descriptor_shap_title"])
    plt.tight_layout()
    bar_fig = (
        "figure_12_microstructural_descriptor_shap_bar_publication_names"
        if target_config["key"] == "Gp"
        else "figure_32_breaking_stress_microstructural_descriptor_shap_bar_publication_names"
    )
    save_fig(bar_fig)
    plt.show()

    return {
        "summary": shap_summary,
        "long": shap_long,
        "shap_values": shap_values,
        "explain": explain,
        "feature_names": desc_display_names,
    }


gpr_descriptor_shap_outputs = {
    target_config["key"]: run_gpr_descriptor_shap(target_config)
    for target_config in TARGET_CONFIGS
}
gpr_descriptor_shap = gpr_descriptor_shap_outputs["Gp"]


def plot_combined_shap_summary(target_config: dict):
    """Combine formulation and structure SHAP summary plots into one 2-row figure."""
    try:
        import shap
    except ImportError:
        print("Combined SHAP summary skipped: install shap with `%pip install shap` and rerun this cell.")
        return

    formulation_payload = gpr_formulation_shap_outputs.get(target_config["key"])
    descriptor_payload = gpr_descriptor_shap_outputs.get(target_config["key"])
    if formulation_payload is None or descriptor_payload is None:
        print(f"Combined SHAP summary skipped for {target_config['key']}: SHAP output is missing.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 11.4))

    plt.sca(axes[0])
    shap.summary_plot(
        formulation_payload["shap_values"],
        formulation_payload["explain"],
        feature_names=formulation_payload["feature_names"],
        show=False,
        plot_size=None,
        max_display=10,
    )
    axes[0].set_title("A. Formulation variables", fontsize=15, fontweight="bold", pad=12)
    axes[0].tick_params(axis="both", labelsize=12)
    axes[0].set_xlabel(target_config["shap_xlabel"], fontsize=13, fontweight="semibold")

    plt.sca(axes[1])
    shap.summary_plot(
        descriptor_payload["shap_values"],
        descriptor_payload["explain"],
        feature_names=descriptor_payload["feature_names"],
        show=False,
        plot_size=None,
        max_display=10,
    )
    axes[1].set_title("B. Microstructural descriptors", fontsize=15, fontweight="bold", pad=12)
    axes[1].tick_params(axis="both", labelsize=12)
    axes[1].set_xlabel(target_config["shap_xlabel"], fontsize=13, fontweight="semibold")

    fig.suptitle(
        f"Combined SHAP summary for {target_config['value_label']}",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.30, right=0.93, bottom=0.07, top=0.93, hspace=0.52)
    combined_fig = (
        "figure_13_combined_formulation_structure_shap_summary_Gp"
        if target_config["key"] == "Gp"
        else "figure_33_combined_formulation_structure_shap_summary_breaking_stress"
    )
    save_fig(combined_fig)
    plt.show()


for target_config in TARGET_CONFIGS:
    plot_combined_shap_summary(target_config)

# %%
# 13. Interpretation text for manuscript

best_row = summary.iloc[0]
print(f"""
Recommended manuscript comparison:

1. Formulation only: tests whether composition and processing variables can
   predict G' without image structure.
2. Structure only: tests whether confocal oil/protein descriptors contain
   sufficient mechanical information.
3. Formulation + structure: tests whether microstructure adds information
   beyond formulation.

All comparisons use stratified grouped cross-validation. Formulation is the
grouping variable, so replicate images from the same formulation are never
split between training and validation. Protein type is the stratification
variable, so folds are balanced to include both FPI and QPI where possible.

Best grouped-CV result in this run:
  {best_row['algorithm']} with {best_row['input_set']}
  R2={best_row['R2_log10']:.3f}, RMSE={best_row['RMSE_log10']:.3f}, MAE={best_row['MAE_log10']:.3f}
""")
