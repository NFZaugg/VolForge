import matplotlib.pyplot as plt
import polars
from matplotlib.axes import Axes

from constants import ASK, BID, MID, STRIKE
from slicer.slice import Slice
from visualization.base_plotter import Plotter


class SlicePlotter(Plotter):
    def _prepare_data(self, slice: Slice) -> polars.DataFrame:
        strikes = sorted(
            set(slice.bid_implied_vols.keys()) | set(slice.ask_implied_vols.keys())
        )
        return (
            polars.DataFrame(
                {
                    STRIKE: strikes,
                    BID: [slice.bid_implied_vols.get(k) for k in strikes],
                    ASK: [slice.ask_implied_vols.get(k) for k in strikes],
                }
            )
            .drop_nulls()
            .sort(STRIKE)
            .with_columns([polars.col(BID) * 100, polars.col(ASK) * 100])
            .with_columns(polars.mean_horizontal(BID, ASK).alias(MID))
        )

    def _draw_spread(self, ax: Axes, k, df):
        ax.fill_between(
            k,
            df[BID],
            df[ASK],
            color=self.palette["spread"],
            alpha=0.15,
            zorder=1,
            label="Bid/Ask Spread",
        )

    def _draw_mid(self, ax: Axes, k, df):
        ax.plot(
            k,
            df[MID],
            color=self.palette["mid"],
            linewidth=0.6,
            linestyle="dashed",
            alpha=0.9,
            zorder=3,
            solid_capstyle="round",
        )

    def _draw_bid_ask(self, ax: Axes, k, df):
        ax.scatter(
            k,
            df[BID],
            marker="o",
            s=32,
            facecolor=self.palette["panel"],
            edgecolor=self.palette["bid"],
            linewidth=1.6,
            label="Bid",
            zorder=4,
        )
        ax.scatter(
            k,
            df[ASK],
            marker="o",
            s=32,
            facecolor=self.palette["panel"],
            edgecolor=self.palette["ask"],
            linewidth=1.6,
            label="Ask",
            zorder=4,
        )

    def _draw_forward(self, ax: Axes, slice: Slice, df):
        ax.axvline(
            slice.forward,
            color=self.palette["forward"],
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            zorder=2,
        )
        ax.text(
            slice.forward,
            df[ASK].max(),
            f"  Fwd {slice.forward:,.2f}",
            color=self.palette["forward"],
            fontsize=9,
            fontweight=400,
            va="bottom",
            ha="left",
        )

    def plot(self, underlying: str, slice: Slice):
        df = self._prepare_data(slice)
        k = df[STRIKE]

        with plt.rc_context(self.rc):
            fig, ax = self._create_figure()

            self._draw_spread(ax, k, df)
            self._draw_mid(ax, k, df)
            self._draw_bid_ask(ax, k, df)
            self._draw_forward(ax, slice, df)
            self._draw_titles(
                ax,
                title=f"{underlying} - Implied Volatility",
                subtitle=f"Expiry {slice.expiry_date}   ·   {(slice.expiry_date - slice.base_date).days} days to maturity",
                xlabel="Strike",
                ylabel="Implied Vol (%)",
            )
            self._style_axes(ax)
            self._draw_legend(ax)

            return self._finalize(fig, ax)
