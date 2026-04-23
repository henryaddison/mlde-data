from collections import defaultdict
import logging
from mlde_utils import DATA_PATH
import xarray as xr

logger = logging.getLogger(__name__)


class PresetSplit:
    """
    Implements a split strategy that returns the dates used by a pre-saved set of splits. The splits are expected to be saved as netCDF files in `DATA_PATH/splits/{preset_name}/{split}.nc`, where `{preset_name}` is the name of the preset (e.g. the name of another dataset). Each netCDF split file should contain a single variable "time" which contains the floor of the time dimension in that split.
    """

    def __init__(
        self,
        preset_name: str,
    ) -> None:
        self.preset_name = preset_name

    def run(self, time_da: xr.DataArray) -> dict[str, xr.DataArray]:
        preset_path = (DATA_PATH / "splits" / self.preset_name).absolute()
        split_paths = preset_path.glob("*.nc")
        splits = defaultdict()
        for split_path in split_paths:
            split_name = split_path.stem
            preset_split_times = xr.open_dataset(split_path)["time"]

            # time_da = time_da["time"].dt.floor("D")
            # split_times = time_da.where(time_da.isin(preset_split_times), drop=True)
            split_times = preset_split_times.where(
                preset_split_times.isin(time_da["time"].dt.floor("D")), drop=True
            )

            splits[split_name] = split_times
        assert len(splits) > 0, f"No splits found for preset {preset_path}"

        return splits
