from abc import ABC

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Plotter(ABC):
    """Base class holding shared plot settings, palette, and generic styling helpers."""

    def __init__(self, figsize: tuple = (10, 6), dark_theme: bool = False):
        self.figsize = figsize
        self.dark_theme = dark_theme
        self.palette = self._build_palette()
        self.rc = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        }

    def _build_palette(self) -> dict:
        if self.dark_theme:
            return {
                "bg": "#0f1117",
                "panel": "#0f1117",
                "ink": "#f1f5f9",
                "muted": "#9ca3af",
                "bid": "#34d399",
                "ask": "#f87171",
                "mid": "#f1f5f9",
                "forward": "#60a5fa",
                "spread": "#64748b",
                "grid": "#27272f",
                "spine": "#3f3f46",
            }
        return {
            "bg": "#fbfbfd",
            "panel": "#fbfbfd",
            "ink": "#1a1a2e",
            "muted": "#6b7280",
            "bid": "#059669",
            "ask": "#dc2626",
            "mid": "#1a1a2e",
            "forward": "#2563eb",
            "spread": "#94a3b8",
            "grid": "#e5e7eb",
            "spine": "#d1d5db",
        }

    def set_dark_theme(self, dark_theme: bool) -> None:
        self.dark_theme = dark_theme
        self.palette = self._build_palette()

    def _create_figure(self) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=self.figsize, facecolor=self.palette["bg"])
        ax.set_facecolor(self.palette["panel"])
        return fig, ax

    def _style_axes(self, ax: Axes) -> None:
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(self.palette["spine"])
        ax.grid(
            True,
            axis="y",
            linestyle="-",
            linewidth=0.6,
            color=self.palette["grid"],
            alpha=0.8,
            zorder=0,
        )
        ax.set_axisbelow(True)

    def _draw_legend(self, ax: Axes, ncol: int = 3) -> None:
        ax.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0, -0.12),
            ncol=ncol,
            fontsize=9.5,
            labelcolor=self.palette["muted"],
            handletextpad=0.5,
            columnspacing=1.5,
        )

    def _draw_titles(
        self, ax: Axes, title: str, subtitle: str, xlabel: str, ylabel: str
    ) -> None:
        ax.set_title(
            title,
            loc="left",
            fontsize=16,
            fontweight="bold",
            color=self.palette["ink"],
            pad=28,
        )
        ax.text(
            0.0,
            1.04,
            subtitle,
            transform=ax.transAxes,
            fontsize=10.5,
            color=self.palette["muted"],
        )
        ax.set_xlabel(xlabel, fontsize=10, color=self.palette["muted"], labelpad=8)
        ax.set_ylabel(ylabel, fontsize=10, color=self.palette["muted"], labelpad=8)
        ax.tick_params(colors=self.palette["muted"], labelsize=9)

    def _finalize(self, fig: Figure, ax: Axes) -> tuple[Figure, Axes]:
        plt.tight_layout()
        return fig, ax
