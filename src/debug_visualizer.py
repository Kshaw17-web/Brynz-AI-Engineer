"""
Debug visualizer for room geometry.
Produces separate plots for:
  1. Scanner trajectory (XZ top-down)
  2. Floor point cloud (XZ)
  3. Detected wall centre-lines
  4. Final room polygon
  5. Combined overlay
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def render_debug(
    xyz: np.ndarray,
    geometry: Dict,
    poses: Optional[Dict],
    output_dir: Path,
    tag: str = "debug",
) -> List[Path]:
    """
    Render debug artifacts. Returns list of paths created.
    Gracefully skips if matplotlib not available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import LineCollection
    except ImportError:
        logger.warning("matplotlib not available; skipping debug visualisation")
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    floor_y  = geometry.get("floor_y", 0.0)
    ceil_y   = geometry.get("ceiling_y", floor_y + 2.5)
    walls    = geometry.get("walls", [])
    polygon  = geometry.get("room_polygon", [])

    # ── Fig 1: Overview with 4 panels ────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor="#0f1117")
    fig.suptitle(f"Geometry Debug — {tag}", color="white", fontsize=13, y=1.01)

    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    def _fmt_ax(ax, title, xlabel="X (m)", ylabel="Z (m)"):
        ax.set_title(title, color="#ccc", fontsize=10)
        ax.set_xlabel(xlabel, color="gray", fontsize=8)
        ax.set_ylabel(ylabel, color="gray", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, color="#222", lw=0.5)

    # Panel 1: Scanner trajectory (from odometry poses)
    ax = axes[0]
    _fmt_ax(ax, "Scanner Trajectory (XZ)")
    if poses is not None and len(poses) > 0:
        pose_list = sorted(poses.items())
        tx = [p["x"] for _, p in pose_list]
        tz = [p["z"] for _, p in pose_list]
        ax.plot(tx, tz, color="#4a9eff", lw=1.0, label="trajectory", alpha=0.8)
        ax.scatter(tx[0], tz[0], color="#22cc44", s=50, zorder=5, label="start")
        ax.scatter(tx[-1], tz[-1], color="#ff4444", s=50, zorder=5, label="end")
        ax.legend(fontsize=7, labelcolor="gray", facecolor="#0f1117", edgecolor="#333")
    else:
        ax.text(0.5, 0.5, "No pose data", transform=ax.transAxes,
                ha="center", color="gray")

    # Panel 2: Floor points (XZ, points with Y near floor)
    ax = axes[1]
    _fmt_ax(ax, f"Floor-layer Points (Y < {floor_y + 0.15:.2f}m)")
    floor_mask = xyz[:, 1] < floor_y + 0.15
    fp = xyz[floor_mask]
    if len(fp) > 0:
        # Subsample for speed
        idx = np.random.choice(len(fp), min(len(fp), 5000), replace=False)
        ax.scatter(fp[idx, 0], fp[idx, 2], s=0.5, c="#88aaff", alpha=0.3)
    ax.text(0.02, 0.98, f"{len(fp)} pts", transform=ax.transAxes,
            color="gray", fontsize=7, va="top")

    # Panel 3: Detected wall centre-lines
    ax = axes[2]
    _fmt_ax(ax, f"Wall Centre-Lines ({len(walls)} walls)")
    WALL_COLORS = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#ff922b", "#cc5de8"]
    for i, w in enumerate(walls):
        s = np.array(w["start_xz"])
        e = np.array(w["end_xz"])
        c = WALL_COLORS[i % len(WALL_COLORS)]
        ax.plot([s[0], e[0]], [s[1], e[1]], color=c, lw=2.0, alpha=0.9)
        mid = (s + e) / 2
        ax.text(mid[0], mid[1], f"{w['length_m']:.1f}m",
                fontsize=6, color=c, ha="center", va="bottom")

    # Panel 4: Room polygon
    ax = axes[3]
    _fmt_ax(ax, "Room Polygon")
    if polygon and len(polygon) >= 3:
        poly = np.array(polygon)
        poly_closed = np.vstack([poly, poly[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], color="#4d96ff", lw=2.5, label="polygon")
        ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, color="#4d96ff")
        for i, (px, pz) in enumerate(poly):
            ax.scatter(px, pz, color="#ffd93d", s=30, zorder=5)
            ax.text(px, pz, f" C{i}", fontsize=6, color="#ffd93d")

        area = geometry.get("floor_area_m2", 0)
        src  = geometry.get("floor_area_source", "?")
        ax.text(0.02, 0.98, f"Area: {area:.2f} m²\nSource: {src}",
                transform=ax.transAxes, color="white", fontsize=7, va="top",
                bbox=dict(boxstyle="round", fc="#0f1117", ec="#333", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No polygon\n(< 3 corners)", transform=ax.transAxes,
                ha="center", color="gray")

    plt.tight_layout()
    out = output_dir / f"{tag}_debug_panels.png"
    fig.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="#0f1117")
    plt.close(fig)
    paths.append(out)
    logger.info(f"  Debug panels → {out}")

    # ── Fig 2: Wall-score table ───────────────────────────────────────────────
    if walls:
        fig2, ax2 = plt.subplots(figsize=(14, max(3, len(walls) * 0.5 + 1.5)),
                                  facecolor="#0f1117")
        ax2.set_facecolor("#0f1117")
        ax2.axis("off")

        headers = ["ID", "Len(m)", "Az(°)", "HtSpan(m)", "HtCov", "RMS(m)", "Density", "Inliers"]
        rows = [[
            w["id"], f"{w['length_m']:.3f}", f"{w['azimuth_deg']:.1f}",
            f"{w['height_span_m']:.3f}", f"{w['height_coverage']:.2f}",
            f"{w['rms_residual_m']:.4f}", f"{w['point_density']:.1f}",
            str(w["n_inliers"])
        ] for w in walls]

        table = ax2.table(
            cellText=rows, colLabels=headers,
            cellLoc="center", loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (r, c), cell in table.get_celld().items():
            cell.set_facecolor("#1a1d27" if r % 2 else "#141720")
            cell.set_edgecolor("#333")
            cell.set_text_props(color="white" if r else "#aaa")

        fig2.suptitle(f"Wall Score Table — {tag}", color="white", fontsize=11)
        out2 = output_dir / f"{tag}_wall_table.png"
        fig2.savefig(str(out2), dpi=120, bbox_inches="tight", facecolor="#0f1117")
        plt.close(fig2)
        paths.append(out2)
        logger.info(f"  Wall table → {out2}")

    return paths
