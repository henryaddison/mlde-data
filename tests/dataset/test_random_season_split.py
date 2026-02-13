import numpy as np
import xarray as xr

import cftime

from mlde_data.dataset.random_season_split import RandomSeasonSplit


def test_split():
    time_range = xr.date_range(
        cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        periods=360 * 20,
        freq="D",
        use_cftime=True,
    )
    time_da = xr.DataArray(dims=["time"], data=time_range, coords={"time": time_range})

    splits = RandomSeasonSplit(
        props={"val": 0.2, "test": 0.1}, time_periods=[["1980-12-01", "2000-12-01"]]
    ).run(time_da)

    # Should divide up by time between splits
    assert len(splits["test"]) == 720
    assert len(splits["val"]) == 1440
    assert len(splits["train"]) == 5040

    # Check that all days are present across the splits
    assert len(np.unique(np.concatenate(list(splits.values())))) == len(time_range)

    # check time is sorted
    for split_times in splits.values():
        assert np.all(split_times[:-1] <= split_times[1:])

    # there should be no overlap between the splits
    assert not any(np.isin(splits["train"], splits["val"]))
    assert not any(np.isin(splits["train"], splits["test"]))
    assert not any(np.isin(splits["val"], splits["test"]))

    # Each split should have a certain number of years for each month
    test_year_seasons = np.unique(
        np.char.add(
            np.vectorize(lambda x: x.year)(splits["test"]).astype("str"),
            np.vectorize(lambda x: x.month)(splits["test"]).astype("str"),
        )
    )
    assert len(test_year_seasons) == 2 * 12

    val_year_seasons = np.unique(
        np.char.add(
            np.vectorize(lambda x: x.year)(splits["val"]).astype("str"),
            np.vectorize(lambda x: x.month)(splits["val"]).astype("str"),
        )
    )
    assert len(val_year_seasons) == 4 * 12

    train_year_seasons = np.unique(
        np.char.add(
            np.vectorize(lambda x: x.year)(splits["train"]).astype("str"),
            np.vectorize(lambda x: x.month)(splits["train"]).astype("str"),
        )
    )
    assert len(train_year_seasons) == 14 * 12


def test_split_hours():
    time_range = xr.date_range(
        cftime.Datetime360Day(1980, 12, 1, 0, 0, 0, 0, has_year_zero=True),
        periods=24 * 360 * 4,
        freq="h",
        use_cftime=True,
    )
    time_da = xr.DataArray(dims=["time"], data=time_range, coords={"time": time_range})

    splits = RandomSeasonSplit(
        props={"val": 0.25, "test": 0.25}, time_periods=[["1980-12-01", "1984-12-01"]]
    ).run(time_da)

    # Should divide up by time between splits
    assert len(splits["test"]) == 360 * 1
    assert len(splits["val"]) == 360 * 1
    assert len(splits["train"]) == 360 * 2

    # Check that all days are present across the splits
    assert np.all(np.isin(time_range.floor("D"), np.concatenate(list(splits.values()))))

    # check time is sorted
    for split_times in splits.values():
        assert np.all(split_times[:-1] <= split_times[1:])

    # there should be no overlap between the splits
    assert not any(np.isin(splits["train"], splits["val"]))
    assert not any(np.isin(splits["train"], splits["test"]))
    assert not any(np.isin(splits["val"], splits["test"]))

    # Each split should have a certain number of years for each month
    test_year_seasons = np.unique(
        np.char.add(
            np.vectorize(lambda x: x.year)(splits["test"]).astype("str"),
            np.vectorize(lambda x: x.month)(splits["test"]).astype("str"),
        )
    )
    assert len(test_year_seasons) == 1 * 12

    val_year_seasons = np.unique(
        np.char.add(
            np.vectorize(lambda x: x.year)(splits["val"]).astype("str"),
            np.vectorize(lambda x: x.month)(splits["val"]).astype("str"),
        )
    )
    assert len(val_year_seasons) == 1 * 12

    train_year_seasons = np.unique(
        np.char.add(
            np.vectorize(lambda x: x.year)(splits["train"]).astype("str"),
            np.vectorize(lambda x: x.month)(splits["train"]).astype("str"),
        )
    )
    assert len(train_year_seasons) == 2 * 12
