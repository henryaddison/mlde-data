"""
Create a new dataset with a variable formed from the sum of the given ones
"""


class Sum:
    def __init__(self, variables, new_variable):
        self.variables = variables
        self.new_variable = new_variable

    def run(self, ds):
        ds = ds.assign(
            {self.new_variable: lambda x: sum([x[var] for var in self.variables])}
        )
        ds = ds.drop_vars(self.variables)
        return ds
