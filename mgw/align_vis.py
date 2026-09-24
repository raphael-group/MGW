"""Tilted two-layer visualisations of MGW alignments."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial import ConvexHull

INK = "#17212B"
MUTED = "#64748B"
MGW_BLUE = "#146C94"
TARGET_ORANGE = "#EA7C2B"
CATEGORY_COLORS = [
    "#146C94", "#E76F51", "#2A9D8F", "#E9C46A", "#7B61A8",
    "#D1495B", "#3A86B7", "#7A9E45", "#F08A5D", "#5B5F97",
    "#00A6A6", "#B56576", "#6D597A", "#577590", "#F4A261",
]
STYLE = {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }


def load_coupling(path: str | Path):
    """Load a coupling without copying dense ``.npy`` arrays into memory."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        return sp.load_npz(path).tocsr()
    raise ValueError(f"Unsupported coupling format: {path}")


def orient_coupling(P, n_rows: int, n_cols: int):
    if P.shape == (n_rows, n_cols):
        return P
    if P.shape == (n_cols, n_rows):
        return P.T
    raise ValueError(f"Coupling shape {P.shape} does not match {(n_rows, n_cols)} in either orientation")


def coupling_marginals(P, block_rows: int = 512) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(P):
        return np.asarray(P.sum(axis=1)).ravel(), np.asarray(P.sum(axis=0)).ravel()
    n_rows, n_cols = P.shape
    row = np.zeros(n_rows, dtype=np.float64)
    col = np.zeros(n_cols, dtype=np.float64)
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        W = np.asarray(P[start:stop], dtype=np.float32)
        row[start:stop] = W.sum(axis=1, dtype=np.float64)
        col += W.sum(axis=0, dtype=np.float64)
    return row, col


def _p_times_matrix(P, X: np.ndarray, block_rows: int = 512) -> np.ndarray:
    if sp.issparse(P):
        return np.asarray(P @ X)
    out = np.zeros((P.shape[0], X.shape[1]), dtype=np.float64)
    for start in range(0, P.shape[0], block_rows):
        stop = min(start + block_rows, P.shape[0])
        out[start:stop] = np.asarray(P[start:stop], dtype=np.float32) @ X
    return out


def coupling_procrustes(X, Y, P, *, ensure_rotation: bool = False):
    """Rigidly map Y into X's frame using only the coupling weights."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    P = orient_coupling(P, len(X), len(Y))
    wx, wy = coupling_marginals(P)
    mass = wx.sum()
    xbar = wx @ X / mass
    ybar = wy @ Y / mass
    Xc, Yc = X - xbar, Y - ybar
    PY = _p_times_matrix(P, Yc)
    H = (Xc.T @ PY).T
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    R = U @ Vt
    if ensure_rotation and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = xbar - ybar @ R
    return Y @ R + t, R, t, wx, wy


def unit_coords(coords, *, flip_y: bool = False) -> np.ndarray:
    xy = np.asarray(coords, dtype=np.float64)[:, :2].copy()
    lo = np.nanpercentile(xy, 0.5, axis=0)
    hi = np.nanpercentile(xy, 99.5, axis=0)
    xy = (xy - lo) / np.maximum(hi - lo, 1e-12)
    xy = np.clip(xy, -0.03, 1.03)
    xy -= np.average(xy, axis=0)
    xy /= max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-12)
    if flip_y:
        xy[:, 1] *= -1
    return xy


def coarsen_to_shared_annotations(labels_top, labels_bottom, *, max_classes: int = 8):
    """Keep abundant annotations found in both layers; collect the rest as Other."""
    a = np.asarray(labels_top, dtype=str)
    b = np.asarray(labels_bottom, dtype=str)
    ca, cb = pd.Series(a).value_counts(), pd.Series(b).value_counts()
    common = ca.index.intersection(cb.index)
    strength = pd.Series({label: min(ca[label], cb[label]) for label in common}).sort_values(ascending=False)
    keep = strength.index[:max_classes].tolist()
    ga = np.where(np.isin(a, keep), a, "Other")
    gb = np.where(np.isin(b, keep), b, "Other")
    classes = keep + (["Other"] if np.any(ga == "Other") or np.any(gb == "Other") else [])
    return ga.astype(str), gb.astype(str), classes


def _category_palette(labels_top, labels_bottom, preferred: Mapping[str, str] | None = None):
    counts = pd.concat([pd.Series(np.asarray(labels_top, dtype=str)), pd.Series(np.asarray(labels_bottom, dtype=str))]).value_counts()
    classes = counts.index.tolist()
    if "Other" in classes:
        classes = [x for x in classes if x != "Other"] + ["Other"]
    palette = {cls: CATEGORY_COLORS[i % len(CATEGORY_COLORS)] for i, cls in enumerate(classes)}
    if "Other" in palette:
        palette["Other"] = "#CBD5E1"
    if preferred:
        palette.update({str(k): v for k, v in preferred.items()})
    return classes, palette


def _add_layer_hull(ax, xy, z, *, color):
    try:
        hull = ConvexHull(xy)
        verts = [[(xy[i, 0], xy[i, 1], z) for i in hull.vertices]]
        patch = Poly3DCollection(
            verts,
            facecolor=mpl.colors.to_rgba(color, 0.018),
            edgecolor=mpl.colors.to_rgba(color, 0.24),
            linewidth=0.7,
        )
        ax.add_collection3d(patch)
    except Exception:
        pass


def _proper_rotation_degrees(R: np.ndarray) -> float:
    """Signed display angle for the row-vector map ``xy @ R``."""
    angle = float(np.degrees(np.arctan2(R[0, 1], R[0, 0])))
    return (angle + 180.0) % 360.0 - 180.0


def _add_procrustes_frames(ax, center, radius, R, *, z_top: float, z_bottom: float) -> None:
    """Draw the fixed top frame and coupling-implied bottom frame on their planes."""
    anchor = np.asarray(center) + np.array([-0.62 * radius, -0.62 * radius])
    length = 0.24 * radius

    def draw_frame(basis, z, color, labels):
        ax.scatter(
            [anchor[0]], [anchor[1]], [z + 0.008],
            s=7, c=[color], linewidths=0, depthshade=False, zorder=8,
        )
        for vector, label in zip(np.asarray(basis), labels):
            vector = vector / max(np.linalg.norm(vector), 1e-12)
            end = anchor + length * vector
            ax.plot(
                [anchor[0], end[0]], [anchor[1], end[1]], [z + 0.008, z + 0.008],
                color=color, linewidth=1.35, alpha=0.96, solid_capstyle="round", zorder=8,
            )
            ax.text(
                end[0], end[1], z + 0.018, label,
                color=color, fontsize=5.8, weight="bold", ha="center", va="center", zorder=9,
            )

    draw_frame(np.eye(2), z_top, MGW_BLUE, ("x", "y"))
    # For row-vector coordinates, the rows of R are the transformed basis axes.
    draw_frame(R, z_bottom, TARGET_ORANGE, ("x′", "y′"))


def _add_layer_badges(ax, top_name: str, bottom_name: str) -> None:
    badge = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4}
    ax.text2D(
        0.045, 0.715, str(top_name), transform=ax.transAxes,
        fontsize=8.1, weight="bold", color=MGW_BLUE, ha="left", va="center", bbox=badge,
    )
    ax.text2D(
        0.045, 0.245, str(bottom_name), transform=ax.transAxes,
        fontsize=8.1, weight="bold", color=TARGET_ORANGE, ha="left", va="center", bbox=badge,
    )


def _wrapped_caption(text: str, figure_width: float) -> str:
    return textwrap.fill(text, width=max(82, int(figure_width * 14.5)))


def _atlas_size(n_panels: int) -> tuple[float, float]:
    """A compact, slide-friendly canvas that still leaves each 3-D panel room."""
    return max(7.0, 3.25 * n_panels), 6.1


def plot_tilted_alignment(
    coords_top,
    coords_bottom,
    labels_top,
    labels_bottom,
    P,
    *,
    dataset_title: str,
    top_name: str,
    bottom_name: str,
    label_description: str,
    save: str | None = None,
    method: str = "MGW",
    k_per_point: int = 3,
    alpha: float | None = None,
    linewidth: float = 0.3,
    line_color: str = INK,
    preferred_colors: Mapping[str, str] | None = None,
    max_points: int = 12000,
    elev: float = 19,
    azim: float = -63,
    seed: int = 7,
    flip_y: bool = False,
    block_rows: int = 512,
):
    """Tilted two-layer view of a coupling: the top-k targets of every source location as alignment lines."""
    with mpl.rc_context(STYLE):
        return _plot_tilted_alignment(**locals())


def _plot_tilted_alignment(
    coords_top, coords_bottom, labels_top, labels_bottom, P, *, dataset_title, top_name, bottom_name,
    label_description, save, method, k_per_point, alpha, linewidth, line_color, preferred_colors,
    max_points, elev, azim, seed, flip_y, block_rows,
):
    A = unit_coords(coords_top, flip_y=flip_y)
    B = unit_coords(coords_bottom, flip_y=flip_y)
    P = load_coupling(P) if isinstance(P, (str, Path)) else P
    P = orient_coupling(P, len(A), len(B))
    B_aligned, R_display, _, _, _ = coupling_procrustes(A, B, P, ensure_rotation=True)
    display_angle = _proper_rotation_degrees(R_display)
    labels_top = np.asarray(labels_top, dtype=str)
    labels_bottom = np.asarray(labels_bottom, dtype=str)
    classes, palette = _category_palette(labels_top, labels_bottom, preferred_colors)
    color_top = np.asarray([palette[x] for x in labels_top])
    color_bottom = np.asarray([palette[x] for x in labels_bottom])
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(A), min(len(A), max_points), replace=False)
    ib = rng.choice(len(B), min(len(B), max_points), replace=False)
    z_top, z_bottom = 0.32, -0.32

    k = int(min(k_per_point, P.shape[1]))
    src, dst, weight = [], [], []
    for start in range(0, P.shape[0], block_rows):
        block = P[start:start + block_rows]
        block = block.toarray() if sp.issparse(block) else np.asarray(block, dtype=np.float64)
        top = np.argpartition(-block, k - 1, axis=1)[:, :k]
        mass = np.take_along_axis(block, top, axis=1)
        peak = block.max(axis=1, keepdims=True)
        src.append(np.repeat(np.arange(start, start + len(block)), k))
        dst.append(top.ravel())
        weight.append((mass / np.maximum(peak, 1e-300)).ravel())
    src, dst, weight = np.concatenate(src), np.concatenate(dst), np.clip(np.concatenate(weight), 0, 1)
    keep = weight > 0
    src, dst, weight = src[keep], dst[keep], weight[keep]
    segments = np.stack([np.c_[A[src], np.full(len(src), z_top)], np.c_[B_aligned[dst], np.full(len(dst), z_bottom)]], axis=1)
    if alpha is None:
        alpha = float(np.clip(300.0 / max(len(src), 1), 0.012, 0.08))
    rgba = np.tile(mpl.colors.to_rgba(line_color), (len(src), 1))
    rgba[:, 3] = alpha * weight

    figure_width, figure_height = _atlas_size(1)
    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.computed_zorder = False
    _add_layer_hull(ax, A, z_top - 0.012, color=MGW_BLUE)
    _add_layer_hull(ax, B_aligned, z_bottom - 0.012, color=TARGET_ORANGE)
    ax.add_collection3d(Line3DCollection(segments, colors=rgba, linewidths=linewidth, zorder=1))
    ax.scatter(A[ia, 0], A[ia, 1], np.full(len(ia), z_top), c=color_top[ia], s=5.0, alpha=0.86, linewidths=0, depthshade=False, rasterized=True, zorder=3)
    ax.scatter(B_aligned[ib, 0], B_aligned[ib, 1], np.full(len(ib), z_bottom), c=color_bottom[ib], s=5.0, alpha=0.74, linewidths=0, depthshade=False, rasterized=True, zorder=2)
    ax.set_title(method, color=MGW_BLUE, weight="bold", fontsize=12.2, pad=0)
    ax.text2D(0.50, 0.900, f"coupling-implied rotation {display_angle:+.0f}°", transform=ax.transAxes, ha="center", fontsize=7.9, color=MUTED)
    ax.text2D(0.50, 0.025, f"{len(src):,} alignment lines  ·  top {k} targets per {top_name} location", transform=ax.transAxes, ha="center", fontsize=7.9, color=MUTED)
    _add_layer_badges(ax, top_name, bottom_name)
    all_xy = np.vstack([A, B_aligned])
    center = all_xy.mean(axis=0)
    radius = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])) * 0.68
    _add_procrustes_frames(ax, center, radius, R_display, z_top=z_top, z_bottom=z_bottom)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(-0.48, 0.48)
    ax.set_box_aspect((1, 1, 0.58), zoom=1.12)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=palette[c], markeredgecolor="none", markersize=7, label=c) for c in classes]
    handles.append(Line2D([0], [0], color=line_color, lw=1.2, alpha=0.6, label="coupling mass"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=min(len(handles), max(4, int(figure_width / 1.4))), fontsize=8.1, handletextpad=0.35, columnspacing=0.85)
    fig.suptitle(dataset_title, x=0.015, y=0.982, ha="left", fontsize=17, weight="bold")
    subtitle = (f"Layers are colored independently by {label_description}; the bottom layer is shown in its "
                "coupling-implied proper-rotation Procrustes frame (no reflection).")
    fig.text(0.015, 0.930, _wrapped_caption(subtitle, figure_width), fontsize=9.1, color=MUTED, va="top", linespacing=1.2)
    footer = (f"Every {top_name} location is joined to its {k} highest-mass targets; line opacity is proportional "
              "to the transported mass relative to that location's peak.")
    fig.text(0.015, 0.102, _wrapped_caption(footer, figure_width), fontsize=8.0, color=MUTED, va="bottom", linespacing=1.25)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.850, bottom=0.205, wspace=-0.06)
    if save is not None:
        for suffix in ("png", "svg"):
            fig.savefig(f"{save}.{suffix}", facecolor="white", bbox_inches="tight", dpi=300)
    return fig
