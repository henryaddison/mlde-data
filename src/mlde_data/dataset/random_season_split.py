import cftime
from collections import defaultdict
import logging
import numpy as np
import xarray as xr

from .base_split import BaseSplit

logger = logging.getLogger(__name__)


class RandomSeasonSplit(BaseSplit):
    def run(self, time_da: xr.DataArray) -> dict[str, list[cftime.Datetime360Day]]:
        rng = np.random.default_rng(seed=self.seed)

        from xarray.groupers import SeasonResampler

        for tp in self.time_periods:
            tp_time_da = time_da.sel(time=slice(tp[0], tp[1]))
            seasons = xr.merge(
                [
                    tp_time_da.resample(
                        time=SeasonResampler(
                            ["DJF", "MAM", "JJA", "SON"], drop_incomplete=False
                        )
                    )
                    .min()
                    .dt.floor("D"),
                    tp_time_da.resample(
                        time=SeasonResampler(
                            ["DJF", "MAM", "JJA", "SON"], drop_incomplete=False
                        )
                    )
                    .max()
                    .dt.ceil("D"),
                ]
            )

            splits = defaultdict(list)
            for season, season_time_ds in seasons.groupby("time.season"):
                nyears = len(season_time_ds["time"])
                p = rng.permutation(nyears)

                split_sizes = {
                    split: int(nyears * prop)
                    for split, prop in self.props.items()
                    if split != "train"
                }
                split_sizes["train"] = nyears - sum(split_sizes.values())

                for split, split_size in split_sizes.items():
                    split_times = np.unique(
                        np.concatenate(
                            [
                                time_da.where(
                                    (time_da.time >= season_time_ds["floor"][idx])
                                    & (time_da.time < season_time_ds["ceil"][idx]),
                                    drop=True,
                                )
                                .dt.floor("D")
                                .values
                                for idx in p[:split_size]
                            ]
                        )
                    )

                    splits[split].append(split_times)
                    p = p[split_size:]
                assert len(p) == 0, "Some times were not assigned to a split"

        return {k: np.sort(np.concatenate(v)) for k, v in splits.items()}

    def _season_start_date(self, season: str, year: int) -> cftime.Datetime360Day:
        if season == "DJF":
            return cftime.Datetime360Day(
                year - 1, 12, 1, 0, 0, 0, 0, has_year_zero=True
            )
        elif season == "MAM":
            return cftime.Datetime360Day(year, 3, 1, 0, 0, 0, 0, has_year_zero=True)
        elif season == "JJA":
            return cftime.Datetime360Day(year, 6, 1, 0, 0, 0, 0, has_year_zero=True)
        elif season == "SON":
            return cftime.Datetime360Day(year, 9, 1, 0, 0, 0, 0, has_year_zero=True)
        else:
            raise ValueError(f"Unknown season: {season}")
