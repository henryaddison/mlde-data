import cftime
import numpy as np
import xarray as xr

from mlde_data.dataset.preset_split import PresetSplit


def test_split():
    # NB time range is longer than time period used for splitting, so some times should be left out of the splits
    time_range = xr.date_range(
        cftime.Datetime360Day(1980, 12, 1, 12, 0, 0, 0, has_year_zero=True),
        periods=360 * 6,
        freq="D",
        use_cftime=True,
    )

    time_da = xr.DataArray(dims=["time"], data=time_range, coords={"time": time_range})

    preset_name = "another-dataset"
    splits = PresetSplit(
        preset_name=preset_name,
    ).run(time_da)

    exp_times = {
        split: xr.date_range(
            cftime.Datetime360Day(start_year, 12, 1, 0, 0, 0, 0, has_year_zero=True),
            periods=360 * duration,
            freq="D",
            use_cftime=True,
        )
        for split, (start_year, duration) in [
            ("train", (1980, 3)),
            ("val", (1983, 1)),
            ("test", (1984, 1)),
        ]
    }

    # Should divide up time
    assert len(splits["test"]) == 360 * 1
    assert len(splits["val"]) == 360 * 1
    assert len(splits["train"]) == 360 * 3

    assert np.all(np.equal(splits["test"]["time"].values, exp_times["test"].values))
    assert np.all(np.equal(splits["val"]["time"].values, exp_times["val"].values))
    assert np.all(np.equal(splits["train"]["time"].values, exp_times["train"].values))

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
        periods=24 * 360 * 5,
        freq="h",
        use_cftime=True,
    )
    time_da = xr.DataArray(dims=["time"], data=time_range, coords={"time": time_range})

    splits = PresetSplit(
        preset_name="another-dataset",
    ).run(time_da)

    exp_times = {
        split: xr.date_range(
            cftime.Datetime360Day(start_year, 12, 1, 0, 0, 0, 0, has_year_zero=True),
            periods=360 * duration,
            freq="D",
            use_cftime=True,
        )
        for split, (start_year, duration) in [
            ("train", (1980, 3)),
            ("val", (1983, 1)),
            ("test", (1984, 1)),
        ]
    }

    # Should divide up time
    assert len(splits["test"]) == 360 * 1
    assert len(splits["val"]) == 360 * 1
    assert len(splits["train"]) == 360 * 3

    assert np.all(np.equal(splits["test"]["time"].values, exp_times["test"].values))
    assert np.all(np.equal(splits["val"]["time"].values, exp_times["val"].values))
    assert np.all(np.equal(splits["train"]["time"].values, exp_times["train"].values))

    # check time is sorted
    for split_times in splits.values():
        assert np.all(split_times[:-1] <= split_times[1:])

    # there should be no overlap between the splits
    assert not any(np.isin(splits["train"], splits["val"]))
    assert not any(np.isin(splits["train"], splits["test"]))
    assert not any(np.isin(splits["val"], splits["test"]))
