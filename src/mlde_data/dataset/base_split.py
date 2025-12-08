import abc
import xarray as xr


class BaseSplit(abc.ABC):
    def __init__(self, props: dict[str, float], seed: int = 42) -> None:
        if "train" not in props:
            props["train"] = 1.0 - sum(props.values())
        self.props = props
        self.seed = seed

    @abc.abstractmethod
    def run(self, combined_dataset: xr.Dataset) -> dict[str, xr.Dataset]: ...
