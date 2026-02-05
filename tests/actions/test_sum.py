import numpy as np
import xarray as xr

from mlde_data.actions import get_action


def test_sum():
    ds = xr.Dataset(
        {"foo": (("x", "y"), [[1, 2], [3, 4]]), "bar": (("x", "y"), [[2, 4], [6, 8]])},
        coords={"x": ["one", "two"], "y": ["a", "b"]},
    )

    summed = get_action("sum")(["foo", "bar"], "baz")(ds)

    expected_values = np.array([[3, 6], [9, 12]])

    assert np.all(summed["baz"].values == expected_values)
