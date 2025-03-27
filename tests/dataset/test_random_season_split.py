import numpy as np
import xarray as xr

import cftime

from mlde_data.dataset.random_season_split import RandomSeasonSplit


def test_split():
    time_range = xr.cftime_range(
        cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        periods=360 * 20,
        freq="D",
    )
    lat_range = np.linspace(-2, 2, 10)
    lon_range = np.linspace(-2, 2, 10)

    ds = xr.Dataset(
        data_vars={
            "measurement": (
                ["time", "grid_longitude", "grid_latitude"],
                np.ones([len(time_range), len(lat_range), len(lon_range)]),
            ),
        },
        coords=dict(
            time=(["time"], time_range),
            grid_longitude=(["grid_longitude"], lon_range),
            grid_latitude=(["grid_latitude"], lat_range),
        ),
    )

    splits = RandomSeasonSplit().run(ds)

    # Should not change the lon and lat but divide up by time
    assert splits["test"]["measurement"].shape == (720, 10, 10)
    assert splits["val"]["measurement"].shape == (1440, 10, 10)
    assert splits["train"]["measurement"].shape == (5040, 10, 10)

    # check time is sorted
    assert np.all(
        splits["test"]["time"].values[:-1] <= splits["test"]["time"].values[1:]
    )
    assert np.all(splits["val"]["time"].values[:-1] <= splits["val"]["time"].values[1:])
    assert np.all(
        splits["train"]["time"].values[:-1] <= splits["train"]["time"].values[1:]
    )

    # there should be no overlap between the splits
    assert not any(np.isin(splits["train"]["time"], splits["val"]["time"]))
    assert not any(np.isin(splits["train"]["time"], splits["test"]["time"]))
    assert not any(np.isin(splits["val"]["time"], splits["test"]["time"]))

    # Each split should have a certain number of years for each month
    test_year_seasons = np.unique(
        np.char.add(
            splits["test"]["time.year"].values.astype("str"),
            splits["test"]["time.month"].values.astype("str"),
        )
    )
    assert len(test_year_seasons) == 2 * 12

    val_year_seasons = np.unique(
        np.char.add(
            splits["val"]["time.year"].values.astype("str"),
            splits["val"]["time.month"].values.astype("str"),
        )
    )
    assert len(val_year_seasons) == 4 * 12

    train_year_seasons = np.unique(
        np.char.add(
            splits["train"]["time.year"].values.astype("str"),
            splits["train"]["time.month"].values.astype("str"),
        )
    )
    assert len(train_year_seasons) == 14 * 12
