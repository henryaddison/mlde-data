from importlib.resources import files
import logging

from mlde_data.actions.actions_registry import register_action
from mlde_data.actions.remapcon import Remapcon
from mlde_data.actions.shift_lon_break import ShiftLonBreak

logger = logging.getLogger(__name__)


@register_action(name="coarsen")
class Coarsen:
    def __init__(self, scale_factor, grid_type=None):
        self.scale_factor = scale_factor
        self.grid_type = grid_type

    def __call__(self, ds):
        logger.info(f"Coarsening by a scale factor of {self.scale_factor}")

        if self.scale_factor == "gcm":
            logger.info(f"Remapping conservatively to gcm grid...")
            # pick the target grid based on the job spec
            # some variables use one grid, others a slightly offset one
            target_grid_filepath = files("mlde_data.actions").joinpath(
                f"target_grids/60km/global/{self.grid_type}/moose_grid.nc"
            )
            ds = Remapcon(target_grid_filepath)(ds)
            ds = ShiftLonBreak()(ds)
            ds = ds.assign_attrs(
                {
                    "resolution": f"{ds.attrs['resolution']}-coarsened-gcm",
                }
            )
        else:
            self.scale_factor = int(self.scale_factor)
            if self.scale_factor == 1:
                logger.info(
                    f"{self.scale_factor}x coarsening scale factor, nothing to do..."
                )
            else:
                logger.info(f"Coarsening {self.scale_factor}x...")
                # horizontally coarsen the hi resolution data
                ds = ds.coarsen(
                    grid_latitude=self.scale_factor,
                    grid_longitude=self.scale_factor,
                    boundary="trim",
                ).mean()

                ds = ds.assign_attrs(
                    {
                        "resolution": f"{ds.attrs['resolution']}-coarsened-{self.scale_factor}x",
                    }
                )

        return ds
