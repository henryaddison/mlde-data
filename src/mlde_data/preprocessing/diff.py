"""
Create a new dataset with a variable formed from the difference of two given
"""


class Diff:
    def __init__(self, left, right, new_variable):
        self.left = left
        self.right = right
        self.new_variable = new_variable

    def run(self, ds):
        ds[self.new_variable] = ds[self.left] - ds[self.right]
        ds = ds.drop_vars([self.left, self.right])
        return ds
