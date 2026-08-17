import matplotlib.pyplot as plt
import numpy as np
import polars
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D

from vol_forge.constants import ASK, BID, MID
from vol_forge.surface.linear_surface import LinearSurface
from vol_forge.visualization.base_plotter import Plotter


class SurfacePlotter(Plotter):
    def _create_figure(self) -> tuple[plt.Figure, Axes3D]:
        fig = plt.figure(figsize=self.figsize, facecolor=self.palette["bg"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(self.palette["panel"])
        return fig, ax

    def _style_axes(self, ax: Axes3D) -> None:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(self.palette["panel"])
            axis.pane.set_edgecolor(self.palette["spine"])
            axis.pane.set_alpha(1.0)
            axis._axinfo["grid"]["color"] = self.palette["grid"]
            axis._axinfo["grid"]["linewidth"] = 0.6

        ax.tick_params(colors=self.palette["muted"], labelsize=8)
        ax.xaxis.label.set_color(self.palette["muted"])
        ax.yaxis.label.set_color(self.palette["muted"])
        ax.zaxis.label.set_color(self.palette["muted"])
        ax.view_init(elev=22, azim=-60)

    def _draw_legend(self, ax: Axes3D, ncol: int = 3) -> None:
        handles = [
            Patch(facecolor=self.palette["mid"], alpha=0.85, label="Mid"),
            Patch(facecolor=self.palette["bid"], alpha=0.35, label="Bid"),
            Patch(facecolor=self.palette["ask"], alpha=0.35, label="Ask Spread"),
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0, -0.05),
            ncol=ncol,
            fontsize=9.5,
            labelcolor=self.palette["muted"],
            handletextpad=0.5,
            columnspacing=1.5,
        )

    def _prepare_grid(
        self, surface: LinearSurface, n_ttm: int = 60, n_strike: int = 60
    ) -> tuple[np.ndarray, np.ndarray, polars.DataFrame]:
        points = np.asarray(surface._mid_interp.points)
        ttms, strikes = points[:, 0], points[:, 1]

        ttm_grid = np.linspace(ttms.min(), ttms.max(), n_ttm)
        strike_grid = np.linspace(strikes.min(), strikes.max(), n_strike)
        T, K = np.meshgrid(ttm_grid, strike_grid)

        mid = surface._mid_interp(T, K) * 100
        bid = surface._bid_interp(T, K) * 100
        ask = surface._ask_interp(T, K) * 100

        return T, K, {MID: mid, BID: bid, ASK: ask}

    def _draw_surface(self, ax: Axes3D, T, K, vols: dict) -> None:
        ax.plot_surface(
            T,
            K,
            vols[MID],
            color=self.palette["mid"],
            alpha=0.85,
            linewidth=0,
            antialiased=True,
            zorder=3,
            shade=False,
        )

    def _draw_spread(self, ax: Axes3D, T, K, vols: dict) -> None:
        for surf, color in (
            (vols[BID], self.palette["bid"]),
            (vols[ASK], self.palette["ask"]),
        ):
            ax.plot_surface(
                T,
                K,
                surf,
                color=color,
                alpha=0.3,
                linewidth=0,
                antialiased=True,
                zorder=1,
            )

    def plot(self, underlying: str, surface: LinearSurface):
        T, K, vols = self._prepare_grid(surface)

        with plt.rc_context(self.rc):
            fig, ax = self._create_figure()

            self._draw_spread(ax, T, K, vols)
            self._draw_surface(ax, T, K, vols)

            self._draw_titles(
                ax,
                title=f"{underlying} - Implied Volatility Surface",
                subtitle=f"As of {surface.base_date}",
                xlabel="TTM",
                ylabel="Strike",
            )
            ax.set_zlabel(
                "Implied Vol (%)", fontsize=10, color=self.palette["muted"], labelpad=8
            )

            self._style_axes(ax)
            self._draw_legend(ax)

            return self._finalize(fig, ax)

    def _draw_titles(
        self, ax: Axes3D, title: str, subtitle: str, xlabel: str, ylabel: str
    ) -> None:
        ax.set_title(
            title,
            loc="left",
            fontsize=16,
            fontweight="bold",
            color=self.palette["ink"],
            pad=28,
        )
        ax.text2D(
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
