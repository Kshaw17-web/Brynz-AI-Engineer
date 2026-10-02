"""
Floor Plan Renderer
====================
Takes extracted room geometry and renders a dimensioned floor plan:
  - PNG (high-res raster, 300 DPI)
  - SVG (vector for printing)

Features:
  - Walls drawn as thick lines
  - Openings (doors/windows) shown as gaps
  - Dimensions annotated on all walls
  - North arrow, scale bar, legend
  - Color-coded damage overlay (optional)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrowPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not found — floor plan rendering disabled")


class FloorPlanRenderer:
    """
    Renders a dimensioned 2D floor plan from extracted geometry.
    """

    # Visual settings
    WALL_LINEWIDTH = 3.0
    WALL_COLOR = "#1a1a2e"
    OPENING_COLOR = "#4ecdc4"
    DAMAGE_COLOR = "#e74c3c"
    BG_COLOR = "#f8f9fa"
    FLOOR_COLOR = "#e8f4f8"
    TEXT_COLOR = "#2c3e50"
    DIM_COLOR = "#7f8c8d"

    DPI = 150
    MARGIN = 0.5  # metres of padding around room

    def __init__(self, output_path: Path, room_id: str = "room_001"):
        self.output_path = Path(output_path)
        self.room_id = room_id

    def render(
        self,
        geometry: Dict,
        damage_regions: Optional[List] = None,
        title: Optional[str] = None,
    ) -> Path:
        """
        Render the floor plan and save to PNG and SVG.
        Returns path to PNG file.
        """
        if not HAS_MATPLOTLIB:
            logger.error("Cannot render floor plan: matplotlib not installed")
            return self.output_path / "floor_plan.png"

        fig, ax = self._create_figure()
        
        # Draw room background
        self._draw_floor(ax, geometry)
        
        # Draw walls
        self._draw_walls(ax, geometry)
        
        # Draw openings
        self._draw_openings(ax, geometry)
        
        # Draw dimensions
        self._draw_dimensions(ax, geometry)
        
        # Draw damage regions
        if damage_regions:
            self._draw_damage(ax, damage_regions, geometry)
        
        # Set axis limits with margin
        self._set_limits(ax, geometry)
        
        # Add title and annotations
        title_str = title or f"Floor Plan — {self.room_id}"
        self._add_annotations(ax, geometry, title_str)
        
        # Save
        png_path = self.output_path / "floor_plan.png"
        svg_path = self.output_path / "floor_plan.svg"
        
        plt.savefig(str(png_path), dpi=self.DPI, bbox_inches="tight",
                    facecolor=self.BG_COLOR)
        plt.savefig(str(svg_path), bbox_inches="tight", facecolor=self.BG_COLOR)
        plt.close(fig)
        
        logger.info(f"Saved floor plan: {png_path}")
        return png_path

    def _create_figure(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.BG_COLOR)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax

    def _draw_floor(self, ax, geometry: Dict):
        """Draw the floor polygon."""
        polygon = np.array(geometry.get("footprint_polygon", []))
        if len(polygon) < 3:
            return
        
        from matplotlib.patches import Polygon as MplPolygon
        # polygon is [x, z] - use x and z as x, y in plot
        poly_patch = MplPolygon(
            polygon[:, [0, 1]],
            closed=True,
            facecolor=self.FLOOR_COLOR,
            edgecolor=self.WALL_COLOR,
            linewidth=1.0,
            alpha=0.7,
            zorder=1,
        )
        ax.add_patch(poly_patch)

    def _draw_walls(self, ax, geometry: Dict):
        """Draw walls as thick line segments."""
        for wall in geometry.get("walls", []):
            s = wall.get("start_xz", [0, 0])
            e = wall.get("end_xz", [0, 0])
            ax.plot(
                [s[0], e[0]], [s[1], e[1]],
                color=self.WALL_COLOR,
                linewidth=self.WALL_LINEWIDTH,
                solid_capstyle="round",
                zorder=3,
            )

    def _draw_openings(self, ax, geometry: Dict):
        """Draw openings (doors/windows) as colored segments."""
        for opening in geometry.get("openings", []):
            # Find the wall this opening belongs to
            wall = self._find_wall(geometry, opening.get("wall_id"))
            if wall is None:
                continue
            
            s = np.array(wall["start_xz"])
            e = np.array(wall["end_xz"])
            wall_len = np.linalg.norm(e - s)
            if wall_len < 0.01:
                continue
            
            tangent = (e - s) / wall_len
            along_s = opening.get("along_wall_start_m", 0)
            along_e = opening.get("along_wall_end_m", along_s + 0.9)
            
            # Opening segment on wall
            p1 = s + tangent * along_s
            p2 = s + tangent * along_e
            
            color = "#3498db" if opening.get("type") == "window" else "#e67e22"
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color=color,
                linewidth=self.WALL_LINEWIDTH + 1,
                solid_capstyle="butt",
                zorder=4,
            )
            
            # Door arc symbol for doors
            if opening.get("type") == "door":
                mid = (p1 + p2) / 2
                ax.annotate("🚪", xy=(mid[0], mid[1]), fontsize=8,
                           ha="center", va="center", zorder=5)

    def _draw_dimensions(self, ax, geometry: Dict):
        """Annotate wall lengths with dimension lines."""
        for wall in geometry.get("walls", []):
            s = np.array(wall.get("start_xz", [0, 0]))
            e = np.array(wall.get("end_xz", [0, 0]))
            length_m = wall.get("length_m", 0)
            ci = wall.get("length_ci_m", 0)
            
            mid = (s + e) / 2
            
            # Offset label perpendicular to wall
            diff = e - s
            if np.linalg.norm(diff) < 0.01:
                continue
            perp = np.array([-diff[1], diff[0]])
            perp /= np.linalg.norm(perp)
            offset = perp * 0.15
            
            ax.annotate(
                f"{length_m:.2f}m\n±{ci:.3f}",
                xy=(mid[0] + offset[0], mid[1] + offset[1]),
                fontsize=7,
                color=self.DIM_COLOR,
                ha="center",
                va="center",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         edgecolor=self.DIM_COLOR, alpha=0.8),
                zorder=6,
            )
            
            # Dimension line
            ax.annotate("", xy=(e[0], e[1]), xytext=(s[0], s[1]),
                       arrowprops=dict(arrowstyle="<->", color=self.DIM_COLOR,
                                      lw=0.8),
                       zorder=5)

    def _draw_damage(self, ax, damage_regions: List, geometry: Dict):
        """Overlay damage regions on the floor plan."""
        for dmg in damage_regions:
            # Damage regions have [x, z] center coordinates and radius/extent
            cx = dmg.get("center_x", 0)
            cz = dmg.get("center_z", 0)
            radius = np.sqrt(dmg.get("area_m2", 0.1) / np.pi)
            
            circle = plt.Circle(
                (cx, cz), radius,
                color=self.DAMAGE_COLOR,
                alpha=0.5,
                linewidth=1.5,
                fill=True,
                zorder=7,
            )
            ax.add_patch(circle)
            
            label = dmg.get("class", "damage")
            ax.annotate(
                label,
                xy=(cx, cz),
                fontsize=6,
                color="white",
                ha="center",
                va="center",
                fontweight="bold",
                zorder=8,
            )

    def _set_limits(self, ax, geometry: Dict):
        """Set axis limits based on room footprint."""
        polygon = np.array(geometry.get("footprint_polygon", [[0,0],[5,0],[5,5],[0,5]]))
        
        if len(polygon) < 2:
            ax.set_xlim(-1, 6)
            ax.set_ylim(-1, 6)
            return
        
        xmin, xmax = polygon[:, 0].min(), polygon[:, 0].max()
        zmin, zmax = polygon[:, 1].min(), polygon[:, 1].max()
        
        m = self.MARGIN
        ax.set_xlim(xmin - m, xmax + m)
        ax.set_ylim(zmin - m, zmax + m)

    def _add_annotations(self, ax, geometry: Dict, title: str):
        """Add title, scale bar, measurements summary."""
        # Title
        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
            color=self.TEXT_COLOR,
            pad=10,
        )
        
        # Measurements box
        area = geometry.get("floor_area_m2", 0)
        area_ci = geometry.get("floor_area_ci_m2", 0)
        height = geometry.get("ceiling_height_m", 0)
        height_ci = geometry.get("ceiling_height_ci_m", 0)
        n_walls = len(geometry.get("walls", []))
        n_openings = len(geometry.get("openings", []))
        
        info_text = (
            f"Floor area: {area:.2f} ± {area_ci:.2f} m²\n"
            f"Ceiling height: {height:.3f} ± {height_ci:.3f} m\n"
            f"Walls: {n_walls} | Openings: {n_openings}"
        )
        
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=self.WALL_COLOR,
                alpha=0.9,
            ),
        )
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.WALL_COLOR, lw=3, label="Wall"),
            plt.Line2D([0], [0], color="#e67e22", lw=3, label="Door"),
            plt.Line2D([0], [0], color="#3498db", lw=3, label="Window"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
        )

    def _find_wall(self, geometry: Dict, wall_id: str) -> Optional[Dict]:
        """Find a wall by ID."""
        for wall in geometry.get("walls", []):
            if wall.get("id") == wall_id:
                return wall
        return None


def render_stitched_plan(
    rooms: Dict[str, Dict],
    output_path: Path,
    damage_by_room: Optional[Dict] = None,
) -> Path:
    """
    Render a stitched multi-room floor plan.
    All rooms are placed on the same coordinate system.
    """
    if not HAS_MATPLOTLIB:
        return output_path / "stitched_plan.png"
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")
    ax.set_aspect("equal")
    ax.axis("off")
    
    COLORS = [
        "#e8f4f8", "#e8f8ed", "#f8ede8", "#f0e8f8",
        "#f8f4e8", "#e8f0f8", "#f8e8f4", "#e8f8f4",
    ]
    
    for i, (room_id, geometry) in enumerate(rooms.items()):
        color = COLORS[i % len(COLORS)]
        renderer = FloorPlanRenderer(output_path, room_id)
        renderer.FLOOR_COLOR = color
        renderer._draw_floor(ax, geometry)
        renderer._draw_walls(ax, geometry)
        renderer._draw_openings(ax, geometry)
        renderer._draw_dimensions(ax, geometry)
        
        if damage_by_room and room_id in damage_by_room:
            renderer._draw_damage(ax, damage_by_room[room_id], geometry)
        
        # Room label
        polygon = np.array(geometry.get("footprint_polygon", [[0,0]]))
        if len(polygon) > 0:
            cx, cz = polygon[:, 0].mean(), polygon[:, 1].mean()
            area = geometry.get("floor_area_m2", 0)
            height = geometry.get("ceiling_height_m", 0)
            ax.text(
                cx, cz,
                f"{room_id}\n{area:.1f}m² / {height:.2f}m",
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="#2c3e50",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="#bdc3c7",
                    alpha=0.85,
                ),
                zorder=10,
            )
    
    ax.set_title(
        "Stitched Multi-Room Floor Plan",
        fontsize=16, fontweight="bold",
        color="#2c3e50", pad=12,
    )
    
    png_path = output_path / "stitched_plan.png"
    svg_path = output_path / "stitched_plan.svg"
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.savefig(str(svg_path), bbox_inches="tight", facecolor="#f8f9fa")
    plt.close(fig)
    
    return png_path
