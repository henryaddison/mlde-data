import abc
import xarray as xr


class BaseSplit(abc.ABC):
    def __init__(
        self,
        props: dict[str, float],
        time_periods: list[list[int]],
        seed: int = 42,
    ) -> None:
        if "train" not in props:
            props["train"] = 1.0 - sum(props.values())
        self.props = props
        self.seed = seed
        self.time_periods = time_periods

    @abc.abstractmethod
    def run(self, combined_dataset: xr.Dataset) -> dict[str, xr.Dataset]: ...
