import numpy as np
import xarray as xr

from ml_downscaling_emulator.preprocessing.sum import Sum


def test_sum():
    ds = xr.Dataset(
        {"foo": (("x", "y"), [[1, 2], [3, 4]]), "bar": (("x", "y"), [[2, 4], [6, 8]])},
        coords={"x": ["one", "two"], "y": ["a", "b"]},
    )

    summed = Sum(["foo", "bar"], "baz").run(ds)

    expected_values = np.array([[3, 6], [9, 12]])

    assert np.all(summed["baz"].values == expected_values)
