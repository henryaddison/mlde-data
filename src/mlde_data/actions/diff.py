"""
Create a new dataset with a variable formed from the difference of two given
"""

import logging
from mlde_data.actions.actions_registry import register_action


@register_action(name="diff")
class Diff:
    def __init__(self, left, right, new_variable):
        self.left = left
        self.right = right
        self.new_variable = new_variable

    def __call__(self, ds):
        logging.info(f"Difference between {self.left} and {self.right}")
        ds[self.new_variable] = ds[self.left] - ds[self.right]
        ds = ds.drop_vars([self.left, self.right])
        return ds
