import numpy as np
import xarray as xr

import cftime

from mlde_data.dataset.random_split import RandomSplit


def test_split():
    time_range = xr.cftime_range(
        cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        periods=360 * 20,
        freq="D",
    )
    time_da = xr.DataArray(dims=["time"], data=time_range, coords={"time": time_range})

    splits = RandomSplit(props={"val": 0.2, "test": 0.1}).run(time_da)

    # Should not change the lon and lat but divide up by time
    assert splits["test"].shape == (720,)
    assert splits["val"].shape == (1440,)
    assert splits["train"].shape == (5040,)

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
