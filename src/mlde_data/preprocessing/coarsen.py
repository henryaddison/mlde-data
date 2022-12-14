import logging

logger = logging.getLogger(__name__)


class Coarsen:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def run(self, ds):
        logger.info(f"Coarsening by a scale factor of {self.scale_factor}")

        # horizontally coarsen the hi resolution data
        coarsened_ds = ds.coarsen(
            grid_latitude=self.scale_factor,
            grid_longitude=self.scale_factor,
            boundary="trim",
        ).mean()

        return coarsened_ds, ds
