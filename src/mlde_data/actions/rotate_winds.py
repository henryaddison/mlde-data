import logging

from mlde_data.actions.actions_registry import register_action
from ncdata.iris_xarray import cubes_from_xarray, cubes_to_xarray
import iris

logger = logging.getLogger(__name__)


@register_action(name="rotate-winds")
class RotateWinds:
    def __init__(self):
        pass

    def __call__(self, ds):
        logger.info(f"Rotating winds to lat/lon grid...")
        grid_mapping_name = ds["x_wind"].attrs["grid_mapping"]
        if grid_mapping_name == "latitude_longitude":
            logger.info("Already on lat/lon grid, nothing to do...")
            return ds
        elif grid_mapping_name == "rotated_latitude_longitude":
            # rotate the winds to the lat/lon grid

            x_cube, y_cube = cubes_from_xarray(ds)

            ds = cubes_to_xarray(
                iris.analysis.cartography.rotate_winds(
                    x_cube, y_cube, x_cube.coord_system().ellipsoid
                )
            )

            return ds
        else:
            raise RuntimeError(f"Unrecognised grid mapping {grid_mapping_name}")
