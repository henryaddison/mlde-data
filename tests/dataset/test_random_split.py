import numpy as np
import xarray as xr

import cftime

from mlde_data.dataset.random_split import RandomSplit


def test_split():
    # NB time range is longer than time period used for splitting, so some times should be left out of the splits
    time_range = xr.date_range(
        cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        periods=360 * 21,
        freq="D",
        use_cftime=True,
    )
    time_da = xr.DataArray(dims=["time"], data=time_range, coords={"time": time_range})

    splits = RandomSplit(
        props={"val": 0.2, "test": 0.1}, time_periods=[["1980-12-01", "2000-11-30"]]
    ).run(time_da)

    # Should divide up time
    assert len(splits["test"]) == 720
    assert len(splits["val"]) == 1440
    assert len(splits["train"]) == 5040

    # check time is sorted
    for split_times in splits.values():
        assert np.all(split_times[:-1] <= split_times[1:])

    # there should be no overlap between the splits
    assert not any(np.isin(splits["train"], splits["val"]))
    assert not any(np.isin(splits["train"], splits["test"]))
    assert not any(np.isin(splits["val"], splits["test"]))


def test_split_hours():
    time_range = xr.date_range(
        cftime.Datetime360Day(1980, 12, 1, 0, 0, 0, 0, has_year_zero=True),
        periods=360 * 3 * 24,
        freq="h",
        use_cftime=True,
    )
    time_da = xr.DataArray(dims=["time"], data=time_range, coords={"time": time_range})

    splits = RandomSplit(
        props={"val": 0.2, "test": 0.1}, time_periods=[["1980-12-01", "1982-11-30"]]
    ).run(time_da)

    # Should divide up by time
    assert len(splits["test"]) == (1 * 36 * 2)
    assert len(splits["val"]) == (2 * 36 * 2)
    assert len(splits["train"]) == (7 * 36 * 2)

    # check time is sorted
    for split_times in splits.values():
        assert np.all(split_times[:-1] <= split_times[1:])

    # there should be no overlap between the splits
    assert not any(np.isin(splits["train"], splits["val"]))
    assert not any(np.isin(splits["train"], splits["test"]))
    assert not any(np.isin(splits["val"], splits["test"]))
