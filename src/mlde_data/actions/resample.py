import datetime
import logging

from mlde_data.actions.actions_registry import register_action

logger = logging.getLogger(__name__)


@register_action(name="resample")
class Resample:
    def __init__(self, target_frequency):
        self.target_frequency = target_frequency
        if target_frequency == "day":
            self.freq = "1D"
            self.offset = datetime.timedelta(hours=12)
        else:
            raise RuntimeError(f"Unknown target frequency {target_frequency}")

    def __call__(self, ds):
        logger.info(f"Resampling to {self.target_frequency}")
        return ds.resample(time=self.freq, loffset=self.offset).mean()
