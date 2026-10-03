"""
Floor Plan Renderer — CP4
===========================
Key CP4 fixes:
  1. Wall segments clipped to room polygon before drawing
  2. Wall start/end computed as intersection of infinite wall line
     with the room polygon boundary
  3. Openings rendered only if `along_wall_start_m` is present
  4. Multi-space support: accepts `spaces` dict from multi-room geometry
  5. Emoji replaced with ASCII (font compatibility)
  6. Axis limits driven by polygon, not raw wall extents
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon as MplPolygon
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not found — floor plan rendering disabled")


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clip_line_to_polygon(
    p1: np.ndarray, p2: np.ndarray, polygon: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Clip an infinite line (defined by points p1, p2) to the interior of a
    convex polygon using the Liang-Barsky parametric method on polygon edges.

    Returns the clipped (start, end) segment, or None if the line does not
    intersect the polygon.
    """
    poly = np.array(polygon, dtype=float)
    if len(poly) < 3:
        return None

    # Ensure polygon is CCW
    area2 = sum(
        (poly[i, 0] * poly[(i+1) % len(poly), 1] -
         poly[(i+1) % len(poly), 0] * poly[i, 1])
        for i in range(len(poly))
    )
    if area2 < 0:
        poly = poly[::-1]

    d = p2 - p1  # direction vector
    t_min, t_max = -1e9, 1e9
    n = len(poly)

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        edge = b - a
        # Outward normal of edge for CCW polygon
        edge_normal = np.array([edge[1], -edge[0]])

        denom = edge_normal @ d
        if abs(denom) < 1e-9:
            # Line parallel to edge — check if outside half-plane
            if edge_normal @ (p1 - a) > 1e-6:
                return None  # strictly outside
            continue

        t = (edge_normal @ (a - p1)) / denom
        if denom < 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)
        if t_min > t_max:
            return None

    if t_min > t_max or (t_max - t_min) < 1e-6:
        return None

    return p1 + t_min * d, p1 + t_max * d


def _wall_to_clipped_segment(
    wall: Dict, polygon: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute a wall's visible segment clipped to the room polygon.

    Uses the wall normal and wall_d (plane distance) to define the infinite
    wall line, then clips it to the polygon.
    """
    if len(polygon) < 3:
        return None

    normal  = np.array(wall.get("normal_xz", [1, 0]), dtype=float)
    tangent = np.array(wall.get("tangent_xz", [0, 1]), dtype=float)
    wall_d  = float(wall.get("wall_d", 0))

    # Point on the wall line: normal * wall_d
    p_on_wall = normal * wall_d

    # Two points defining the infinite line
    p1 = p_on_wall - tangent * 50.0
    p2 = p_on_wall + tangent * 50.0

    # Ensure polygon is CCW for consistent clipping
    poly = np.array(polygon, dtype=float)
    area2 = sum(
        (poly[i, 0] * poly[(i+1) % len(poly), 1] -
         poly[(i+1) % len(poly), 0] * poly[i, 1])
        for i in range(len(poly))
    )
    if area2 < 0:
        poly = poly[::-1]  # flip to CCW

    return _clip_line_to_polygon(p1, p2, poly)


def _polygon_area(polygon: np.ndarray) -> float:
    n = len(polygon)
    if n < 3:
        return 0.0
    area = sum(
        polygon[i, 0] * polygon[(i+1) % n, 1] -
        polygon[(i+1) % n, 0] * polygon[i, 1]
        for i in range(n)
    )
    return abs(area) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Main renderer
# ─────────────────────────────────────────────────────────────────────────────

class FloorPlanRenderer:
    """Renders a dimensioned 2D floor plan from extracted geometry."""

    WALL_LINEWIDTH  = 3.0
    WALL_COLOR      = "#1a1a2e"
    OPENING_COLOR   = "#e67e22"
    WINDOW_COLOR    = "#3498db"
    CORNER_COLOR    = "#e74c3c"
    BG_COLOR        = "#f8f9fa"
    FLOOR_COLOR     = "#e8f4f8"
    TEXT_COLOR      = "#2c3e50"
    DIM_COLOR       = "#7f8c8d"

    DPI    = 150
    MARGIN = 0.5  # metres of padding around room

    def __init__(self, output_path: Path, room_id: str = "room_001"):
        self.output_path = Path(output_path)
        self.room_id = room_id

    def render(
        self,
        geometry: Dict,
        damage_regions: Optional[List] = None,
        title: Optional[str] = None,
        space_label: Optional[str] = None,
    ) -> Path:
        """Render and save floor plan. Returns PNG path."""
        if not HAS_MATPLOTLIB:
            logger.error("Cannot render floor plan: matplotlib not installed")
            return self.output_path / "floor_plan.png"

        polygon_raw = geometry.get("room_polygon") or geometry.get("footprint_polygon", [])
        polygon = np.array(polygon_raw, dtype=float) if len(polygon_raw) >= 3 else np.zeros((0, 2))

        fig, ax = self._create_figure()
        self._draw_floor(ax, polygon)
        self._draw_walls_clipped(ax, geometry, polygon)
        self._draw_openings(ax, geometry)
        self._draw_corners(ax, polygon)
        if damage_regions:
            self._draw_damage(ax, damage_regions)
        self._set_limits(ax, polygon)
        title_str = title or f"Floor Plan — {space_label or self.room_id}"
        self._add_annotations(ax, geometry, title_str, polygon)

        png_path = self.output_path / "floor_plan.png"
        svg_path = self.output_path / "floor_plan.svg"
        plt.savefig(str(png_path), dpi=self.DPI, bbox_inches="tight", facecolor=self.BG_COLOR)
        plt.savefig(str(svg_path), bbox_inches="tight", facecolor=self.BG_COLOR)
        plt.close(fig)
        logger.info(f"Saved floor plan: {png_path}")
        return png_path

    # ── Drawing helpers ──────────────────────────────────────────────────────

    def _create_figure(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.BG_COLOR)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax

    def _draw_floor(self, ax, polygon: np.ndarray):
        if len(polygon) < 3:
            return
        poly_patch = MplPolygon(
            polygon[:, [0, 1]], closed=True,
            facecolor=self.FLOOR_COLOR, edgecolor="none",
            alpha=0.85, zorder=1,
        )
        ax.add_patch(poly_patch)

    def _draw_walls_clipped(self, ax, geometry: Dict, polygon: np.ndarray):
        """
        Draw only the portion of each wall that lies inside the room polygon.
        Clips infinite wall lines to the polygon boundary.
        Stores the clipped segment back into wall as 'render_start_xz' / 'render_end_xz'.
        """
        for wall in geometry.get("walls", []):
            seg = _wall_to_clipped_segment(wall, polygon)
            if seg is None:
                # Fall back to raw start/end if clipping fails
                s = np.array(wall.get("start_xz", [0, 0]))
                e = np.array(wall.get("end_xz", [0, 0]))
                # Only draw if within 2× polygon bbox
                if len(polygon) >= 3:
                    bbox = polygon.max(axis=0) - polygon.min(axis=0)
                    if np.linalg.norm(e - s) > max(bbox) * 2:
                        continue
                seg = (s, e)

            s, e = seg
            # Store clipped segment for opening rendering and dimension label
            wall["render_start_xz"] = s.tolist()
            wall["render_end_xz"]   = e.tolist()
            wall["render_length_m"] = float(np.linalg.norm(e - s))

            ax.plot(
                [s[0], e[0]], [s[1], e[1]],
                color=self.WALL_COLOR,
                linewidth=self.WALL_LINEWIDTH,
                solid_capstyle="round",
                zorder=3,
            )

            # Dimension label at midpoint
            mid = (s + e) / 2
            diff = e - s
            if np.linalg.norm(diff) < 0.1:
                continue
            perp = np.array([-diff[1], diff[0]])
            perp /= np.linalg.norm(perp)
            offset = perp * 0.18
            length_m  = wall.get("length_m", np.linalg.norm(e - s))
            ci        = wall.get("length_ci_m", 0)
            ax.annotate(
                f"{length_m:.2f}m\n+/-{ci:.3f}",
                xy=(mid[0] + offset[0], mid[1] + offset[1]),
                fontsize=7, color=self.DIM_COLOR,
                ha="center", va="center", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=self.DIM_COLOR, alpha=0.85),
                zorder=6,
            )

    def _draw_openings(self, ax, geometry: Dict):
        """
        Draw openings on the clipped wall segments.
        Requires `along_wall_start_m` and `along_wall_end_m` in each opening.
        If these fields are absent, the opening is not rendered (we do not
        fabricate position on wall).
        """
        wall_map = {w["id"]: w for w in geometry.get("walls", [])}

        for opening in geometry.get("openings", []):
            wall = wall_map.get(opening.get("wall_id"))
            if wall is None:
                continue

            # Use rendered (clipped) start, fallback to raw
            s = np.array(wall.get("render_start_xz") or wall.get("start_xz", [0, 0]))
            e = np.array(wall.get("render_end_xz")   or wall.get("end_xz",   [0, 0]))
            wall_len = float(np.linalg.norm(e - s))
            if wall_len < 0.01:
                continue

            tangent = (e - s) / wall_len

            # Use explicit along_wall positions if present
            a_start = opening.get("along_wall_start_m")
            a_end   = opening.get("along_wall_end_m")

            if a_start is None or a_end is None:
                # No position information → draw marker at wall centre
                # with width annotation only — don't fabricate position
                mid = (s + e) / 2
                w_m = opening.get("width_m", 0)
                color = self.WINDOW_COLOR if opening.get("type") == "window" else self.OPENING_COLOR
                ax.scatter(mid[0], mid[1], s=80, c=color,
                           marker="D", zorder=5, alpha=0.7)
                ax.annotate(
                    f"[?] {opening['type']} ~{w_m:.1f}m\n(pos unverified)",
                    xy=(mid[0], mid[1]), fontsize=6,
                    color=color, ha="center", va="bottom",
                    xytext=(0, 12), textcoords="offset points",
                    zorder=6,
                )
                continue

            p1 = s + tangent * float(a_start)
            p2 = s + tangent * float(a_end)

            color = self.WINDOW_COLOR if opening.get("type") == "window" else self.OPENING_COLOR
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color=color, linewidth=self.WALL_LINEWIDTH + 2,
                solid_capstyle="butt", zorder=4,
            )
            mid = (p1 + p2) / 2
            label = "W" if opening.get("type") == "window" else "D"
            ax.annotate(
                label,
                xy=(mid[0], mid[1]), fontsize=8,
                ha="center", va="center", color="white",
                fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.2", facecolor=color,
                          edgecolor="none"),
                zorder=5,
            )

    def _draw_corners(self, ax, polygon: np.ndarray):
        if len(polygon) < 3:
            return
        # Draw polygon outline
        poly_closed = np.vstack([polygon, polygon[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1],
                color=self.WALL_COLOR, lw=1.5, ls="--", zorder=2, alpha=0.5)
        # Corner dots
        ax.scatter(polygon[:, 0], polygon[:, 1],
                   s=40, c=self.CORNER_COLOR, zorder=7)
        for i, (px, pz) in enumerate(polygon):
            ax.annotate(f"C{i}", xy=(px, pz), fontsize=7,
                        color=self.CORNER_COLOR, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points", zorder=8)

    def _draw_damage(self, ax, damage_regions: List):
        for dmg in damage_regions:
            cx = dmg.get("center_x", 0)
            cz = dmg.get("center_z", 0)
            r  = np.sqrt(max(0.01, dmg.get("area_m2", 0.1)) / np.pi)
            circle = plt.Circle((cx, cz), r, color="#e74c3c",
                                  alpha=0.4, zorder=7)
            ax.add_patch(circle)

    def _set_limits(self, ax, polygon: np.ndarray):
        if len(polygon) >= 3:
            xmin, xmax = polygon[:, 0].min(), polygon[:, 0].max()
            zmin, zmax = polygon[:, 1].min(), polygon[:, 1].max()
        else:
            xmin, xmax, zmin, zmax = -1, 6, -1, 6
        m = self.MARGIN
        ax.set_xlim(xmin - m, xmax + m)
        ax.set_ylim(zmin - m, zmax + m)

    def _add_annotations(self, ax, geometry: Dict, title: str, polygon: np.ndarray):
        ax.set_title(title, fontsize=14, fontweight="bold",
                     color=self.TEXT_COLOR, pad=10)

        area    = geometry.get("floor_area_m2", 0)
        area_ci = geometry.get("floor_area_ci_m2", 0)
        height  = geometry.get("ceiling_height_m", 0)
        h_ci    = geometry.get("ceiling_height_ci_m", 0)
        src     = geometry.get("floor_area_source", "?")
        h_ok    = geometry.get("ceiling_detection_reliable", "?")
        n_w     = len(geometry.get("walls", []))
        n_o     = len(geometry.get("openings", []))
        n_c     = len(polygon)
        perim   = geometry.get("room_perimeter_m", 0)

        info_text = (
            f"Floor area:  {area:.2f} +/- {area_ci:.2f} m2 [{src}]\n"
            f"Perimeter:   {perim:.2f} m   Corners: {n_c}\n"
            f"Ceiling:     {height:.3f} +/- {h_ci:.3f} m [reliable:{h_ok}]\n"
            f"Walls: {n_w}   Openings: {n_o} [low-confidence]"
        )
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=self.WALL_COLOR, alpha=0.92),
        )

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.WALL_COLOR, lw=3, label="Wall (clipped)"),
            plt.Line2D([0], [0], color=self.OPENING_COLOR, lw=3, label="Door (unverified)"),
            plt.Line2D([0], [0], color=self.WINDOW_COLOR, lw=3, label="Window (unverified)"),
            plt.Line2D([0], [0], color=self.CORNER_COLOR, lw=0,
                       marker="o", markersize=6, label="Corner"),
        ]
        ax.legend(handles=legend_elements, loc="lower right",
                  fontsize=8, framealpha=0.92)

    def _find_wall(self, geometry: Dict, wall_id: str):
        for wall in geometry.get("walls", []):
            if wall.get("id") == wall_id:
                return wall
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Multi-space plan renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_multi_space_plan(
    spaces: Dict[str, Dict],
    output_path: Path,
    title: str = "Multi-Space Floor Plan",
) -> Path:
    """
    Render all detected spaces on a single canvas.
    `spaces` is a dict of space_id → geometry dict.
    """
    if not HAS_MATPLOTLIB:
        return output_path / "floor_plan.png"

    SPACE_COLORS = [
        "#dce8f0", "#dcefd8", "#f0dce8", "#f0eadc",
        "#dce8f0", "#e8f0dc", "#f0dce0", "#dcf0ea",
    ]

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")
    ax.set_aspect("equal")
    ax.axis("off")

    all_pts = []
    for idx, (space_id, geom) in enumerate(spaces.items()):
        color = SPACE_COLORS[idx % len(SPACE_COLORS)]
        polygon_raw = geom.get("room_polygon") or geom.get("footprint_polygon", [])
        polygon = np.array(polygon_raw, dtype=float) if len(polygon_raw) >= 3 else np.zeros((0, 2))

        r = FloorPlanRenderer(output_path, space_id)
        r.FLOOR_COLOR = color
        r._draw_floor(ax, polygon)
        r._draw_walls_clipped(ax, geom, polygon)
        r._draw_openings(ax, geom)
        r._draw_corners(ax, polygon)

        if len(polygon) >= 3:
            all_pts.append(polygon)
            cx, cz = polygon[:, 0].mean(), polygon[:, 1].mean()
            area   = geom.get("floor_area_m2", 0)
            h      = geom.get("ceiling_height_m", 0)
            n_w    = len(geom.get("walls", []))
            ax.text(cx, cz,
                    f"{space_id}\n{area:.1f}m2  h={h:.2f}m\n{n_w} walls",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#bdc3c7", alpha=0.88),
                    zorder=10)

    if all_pts:
        pts = np.vstack(all_pts)
        m = 0.5
        ax.set_xlim(pts[:, 0].min() - m, pts[:, 0].max() + m)
        ax.set_ylim(pts[:, 1].min() - m, pts[:, 1].max() + m)
    else:
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)

    ax.set_title(title, fontsize=16, fontweight="bold", color="#2c3e50", pad=12)

    png_path = output_path / "floor_plan.png"
    svg_path = output_path / "floor_plan.svg"
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.savefig(str(svg_path), bbox_inches="tight", facecolor="#f8f9fa")
    plt.close(fig)
    logger.info(f"Saved multi-space floor plan: {png_path}")
    return png_path


# Backwards-compatibility alias
def render_stitched_plan(rooms, output_path, damage_by_room=None):
    return render_multi_space_plan(rooms, Path(output_path))
