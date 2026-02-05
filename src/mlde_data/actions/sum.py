"""
Create a new dataset with a variable formed from the sum of the given ones
"""

import logging
from mlde_data.actions.actions_registry import register_action


@register_action(name="sum")
class Sum:
    def __init__(self, variables, new_variable):
        self.variables = variables
        self.new_variable = new_variable

    def __call__(self, ds):
        logging.info(f"Summing {self.variables}")
        ds = ds.assign(
            {self.new_variable: lambda x: sum([x[var] for var in self.variables])}
        )

        return ds
