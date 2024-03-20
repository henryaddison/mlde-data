import abc
import xarray as xr


class BaseSplit(abc.ABC):
    def __init__(
        self, val_prop: float = 0.2, test_prop: float = 0.1, seed: int = 42
    ) -> None:
        self.val_prop = val_prop
        self.test_prop = test_prop
        self.seed = seed

    @abc.abstractmethod
    def run(self, combined_dataset: xr.Dataset) -> dict[str, xr.Dataset]:
        ...
