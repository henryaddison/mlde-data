import logging

logger = logging.getLogger(__name__)


class SelectGCMDomain:

    DOMAIN_CENTRES = {
        "2.2km-coarsened-4x_bham-64": dict(
            longitude=slice(209, 218),
            latitude=slice(252, 261),
        ),
    }

    def __init__(self, subdomain) -> None:
        self.subdomain = subdomain

    def run(self, ds):
        logger.info(f"Selecting GCM subdomain {self.subdomain}")

        return ds.isel(**self.DOMAIN_CENTRES[self.subdomain])
