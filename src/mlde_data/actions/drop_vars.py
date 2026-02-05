import logging

from mlde_data.actions.actions_registry import register_action

logger = logging.getLogger(__name__)


@register_action(name="drop-variables")
class Rename:
    def __init__(self, variables):
        self.variables = variables

    def __call__(self, ds):
        logger.info(f"Dropping variables {self.variables}")

        return ds.drop_vars(self.variables)
