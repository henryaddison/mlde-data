import logging

from mlde_data.actions.actions_registry import register_action

logger = logging.getLogger(__name__)


@register_action(name="rename")
class Rename:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, ds):
        logger.info(f"Renaming {self.mapping}")

        return ds.rename(self.mapping)
