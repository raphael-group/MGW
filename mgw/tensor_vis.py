"""
tensor_vis.py — Riemannian deformation-grid visualisation for MGW.

Public API
----------
plot_deformation_grid(ax, coords, W, G_tensors, label, **kwargs)
    Draw the Riemannian deformation grid on *ax*.  Grid lines are level sets
    of two harmonic scalar fields (u: left→right, v: bottom→top) solved on
    the Riemannian kNN graph W via the Laplace-Beltrami operator.  The
    background heatmap shows local metric anisotropy log(λ_max / λ_min).

Helper functions (also importable)
-----------------------------------
metric_to_grid   – scatter per-spot pullback metrics onto a regular grid.
harmonic_coords  – solve Δ_g u = 0 with Dirichlet BCs on a sparse graph.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors


def metric_to_grid(coords, G_tensors, nx=100, ny=100, sigma=2.0, pad=0.08):
    """
    Interpolate per-spot pullback metrics onto a regular grid and Gaussian-smooth.

    Parameters
    ----------
    coords    : (N, 2) float array of spatial coordinates.
    G_tensors : (N, 2, 2) tensor or array of pullback metric matrices.
    nx, ny    : grid resolution.
    sigma     : Gaussian smoothing sigma (pixels) applied to each component.
    pad       : fractional margin added on each side so the tissue is not flush
                against the grid boundary (avoids edge clipping in the heatmap).

    Returns
    -------
    gx, gy   : 1-D coordinate arrays of the padded grid.
    G_grid   : (ny, nx, 2, 2) smoothed metric field.
    tmask    : (ny, nx) bool tissue mask.
    """
    G = G_tensors.detach().cpu().numpy() if hasattr(G_tensors, 'detach') else np.asarray(G_tensors, float)
    x, y = coords[:, 0], coords[:, 1]
    dx, dy = (x.max() - x.min()) * pad, (y.max() - y.min()) * pad
    gx = np.linspace(x.min() - dx, x.max() + dx, nx)
    gy = np.linspace(y.min() - dy, y.max() + dy, ny)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.c_[x, y]
    G_grid = np.zeros((ny, nx, 2, 2))
    for i in range(2):
        for j in range(2):
            v = G[:, i, j]
            z = griddata(pts, v, (GX, GY), method='linear')
            bad = np.isnan(z)
            if bad.any():
                z[bad] = griddata(pts, v, (GX[bad], GY[bad]), method='nearest')
            G_grid[:, :, i, j] = gaussian_filter(z, sigma=sigma)
    nn = NearestNeighbors(n_neighbors=1).fit(pts)
    d, _ = nn.kneighbors(np.c_[GX.ravel(), GY.ravel()])
    spacing = max((x.max() - x.min()) / nx, (y.max() - y.min()) / ny) * 2.5
    tmask = (d[:, 0] < spacing).reshape(ny, nx)
    return gx, gy, G_grid, tmask


def harmonic_coords(W_arc, coords, axis='x', bnd_pct=5):
    """
    Solve Δ_g u = 0 on the Riemannian kNN graph with Dirichlet boundary conditions.

    Parameters
    ----------
    W_arc   : scipy sparse matrix with Riemannian arc lengths as edge weights
              (as returned by plotting.build_weighted_graph).
    coords  : (N, 2) spatial coordinates.
    axis    : 'x' → u=0 at left edge, u=1 at right edge;
              'y' → v=0 at bottom, v=1 at top.
    bnd_pct : percentile used to define the two Dirichlet boundary strips.

    Returns
    -------
    u : (N,) array in [0, 1].  Level sets of u (and of v) are the deformation
        grid lines — metric-orthogonal and tissue-spanning by construction.
    """
    W = W_arc.copy().astype(float)
    W.data = 1.0 / (W.data + 1e-12)   # arc lengths → conductances
    W = W.tocsr()
    n = W.shape[0]
    vals = coords[:, 0] if axis == 'x' else coords[:, 1]
    lo = vals <= np.percentile(vals, bnd_pct)
    hi = vals >= np.percentile(vals, 100 - bnd_pct)
    interior = ~(lo | hi)
    int_idx = np.where(interior)[0]
    bnd_lo  = np.where(lo)[0]
    bnd_hi  = np.where(hi)[0]
    bnd_idx = np.concatenate([bnd_lo, bnd_hi])
    u_bnd   = np.concatenate([np.zeros(len(bnd_lo)), np.ones(len(bnd_hi))])
    d = np.asarray(W.sum(axis=1)).ravel()
    L = diags(d) - W
    L_ii = L.tocsr()[int_idx][:, int_idx]
    L_ib = L.tocsr()[int_idx][:, bnd_idx]
    u_int = spsolve(L_ii.tocsc(), -(L_ib @ u_bnd))
    u = np.zeros(n)
    u[int_idx] = np.clip(u_int, 0.0, 1.0)
    u[bnd_lo]  = 0.0
    u[bnd_hi]  = 1.0
    return u


def _smooth_with_nans(arr, sigma):
    """Gaussian-smooth a 2-D field that contains NaNs, preserving the NaN boundary."""
    nan_mask = np.isnan(arr)
    s_num = gaussian_filter(np.where(nan_mask, 0.0, arr), sigma=sigma)
    s_den = gaussian_filter((~nan_mask).astype(float), sigma=sigma)
    with np.errstate(invalid='ignore'):
        out = np.where(s_den > 0.05, s_num / s_den, np.nan)
    out[nan_mask] = np.nan
    return out


def plot_deformation_grid(ax, coords, W, G_tensors, label,
                          nx=120, ny=120, sigma_smooth=2.0,
                          n_lines=36, smooth_contour=2.0,
                          lw=0.65, line_alpha=0.90,
                          line_color='#1c1c1c',
                          bg_cmap='gray_r', alpha_bg=0.72,
                          bnd_pct=5, pad=0.08):
    """
    Draw a Riemannian deformation grid on *ax*.

    Grid lines are level sets of two harmonic coordinate fields u (left→right)
    and v (bottom→top), solved on the Riemannian kNN graph W via the discrete
    Laplace-Beltrami operator.  Background shows log(λ_max / λ_min).

    Parameters
    ----------
    ax         : matplotlib Axes to draw on.
    coords     : (N, 2) spatial coordinates (e.g. xs or xs2).
    W          : scipy sparse arc-length–weighted kNN graph (W_M or W_N).
    G_tensors  : (N, 2, 2) pullback metric tensors (G_M or G_N).
    label      : axes title string.
    nx, ny     : resolution of the background metric grid.
    sigma_smooth : Gaussian σ for smoothing the metric field before eigendecomposition.
    n_lines    : number of contour levels per coordinate family.
    smooth_contour : Gaussian σ (grid pixels) applied to u/v before contouring —
                     removes waviness from the discrete graph.
    lw         : contour line width.
    line_alpha : contour line opacity.
    line_color : contour line colour.
    bg_cmap    : colormap for the anisotropy heatmap ('gray_r' for grayscale).
    alpha_bg   : heatmap opacity.
    bnd_pct    : percentile defining the Dirichlet boundary strips.
    pad        : fractional margin built into the grid on each side — the heatmap
                 and contours are computed over this padded region so the tissue
                 is never flush against the plot boundary.

    Returns
    -------
    im : AxesImage (the heatmap), suitable for a colorbar.
    """
    # ── 1. Anisotropy heatmap ─────────────────────────────────────────────
    gx, gy, G_grid, tmask = metric_to_grid(
        coords, G_tensors, nx=nx, ny=ny, sigma=sigma_smooth, pad=pad)
    G_flat = G_grid.reshape(-1, 2, 2)
    lam, _ = np.linalg.eigh(G_flat)
    ny_, nx_ = G_grid.shape[:2]
    log_aniso = gaussian_filter(
        np.log(np.maximum(lam[:, 1], 1e-12) / np.maximum(lam[:, 0], 1e-12)
               ).reshape(ny_, nx_), sigma=1.5)
    log_aniso[~tmask] = np.nan

    vmax_a = float(np.nanpercentile(log_aniso, 95)) or 1.0
    im = ax.imshow(log_aniso, origin='lower',
                   extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                   cmap=bg_cmap, alpha=alpha_bg, aspect='equal',
                   vmin=0, vmax=vmax_a, zorder=1)

    # ── 2. Harmonic coordinates ───────────────────────────────────────────
    print(f'  Solving harmonic coords for {label}...')
    u = harmonic_coords(W, coords, axis='x', bnd_pct=bnd_pct)
    v = harmonic_coords(W, coords, axis='y', bnd_pct=bnd_pct)

    # ── 3. Interpolate onto the padded grid ───────────────────────────────
    GCX, GCY = np.meshgrid(gx, gy)
    nn_c = NearestNeighbors(n_neighbors=1).fit(coords)
    dc, _ = nn_c.kneighbors(np.c_[GCX.ravel(), GCY.ravel()])
    sp = max((coords[:, 0].max() - coords[:, 0].min()) / nx,
             (coords[:, 1].max() - coords[:, 1].min()) / ny) * 4.0
    cmask = (dc[:, 0] < sp).reshape(ny_, nx_)

    u_grid = griddata(coords, u, (GCX, GCY), method='linear')
    v_grid = griddata(coords, v, (GCX, GCY), method='linear')
    u_grid[~cmask] = np.nan
    v_grid[~cmask] = np.nan

    u_grid = _smooth_with_nans(u_grid, sigma=smooth_contour)
    v_grid = _smooth_with_nans(v_grid, sigma=smooth_contour)

    # ── 4. Contours ───────────────────────────────────────────────────────
    levels = np.linspace(0.05, 0.95, n_lines)
    ckw = dict(colors=line_color, linewidths=lw, alpha=line_alpha, zorder=3)
    ax.contour(GCX, GCY, u_grid, levels=levels, **ckw)
    ax.contour(GCX, GCY, v_grid, levels=levels, **ckw)

    ax.set_xlim(gx.min(), gx.max())
    ax.set_ylim(gy.min(), gy.max())
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(label, fontsize=13, fontweight='bold')
    return im
