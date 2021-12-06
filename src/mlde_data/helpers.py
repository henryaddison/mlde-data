import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

# Our two different projections
cp_model_rotated_pole = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
platecarree = ccrs.PlateCarree()

def plot_with_ts(datasets, timestamps, variable='pr', vmin=0, vmax=None, cmap='YlGn', titles=None):
    figs = []
    for t, timestamp in enumerate(timestamps):
        f, axes = plt.subplots(1, len(datasets), figsize=(30, 6), subplot_kw={'projection': cp_model_rotated_pole})
        f.tight_layout(h_pad=2)
        # make sure axes is 2-d even if only 1 timestamp and or slice
        if len(datasets) == 1: axes = [axes]

        for i, data in enumerate(datasets):
            ax = axes[i]
            ax.coastlines()

            x = "longitude"
            y = "latitude"
            transform = platecarree
            if "grid_latitude" in data.coords.keys():
                x = f"grid_longitude"
                y = f"grid_latitude"
                transform = cp_model_rotated_pole

            data.sel(time=timestamp)[variable].plot(ax=ax, x=x, y=y, add_colorbar=True, transform = transform, vmin=vmin, vmax=vmax, cmap=cmap)
            if titles:
                ax.set_title(f"{titles[i]}@{timestamp.values}")

        figs.append(f)
#     plt.show()
    return figs

MODEL2RES = {
    "gcm": "60km",
    "cpm": "2.2km"
}

def data_filepath(horizontal_desc, source_model, variable, year, rcp="rcp85", ensemble_member="01", temp_res="day"):
    source_res = MODEL2RES[source_model]
    year_range = f"{year}1201-{year+1}1130"

    file_name = f"{variable}_{rcp}_land-{source_model}_uk_{source_res}_{ensemble_member}_{temp_res}_{year_range}.nc"
    if horizontal_desc == "2.2km":
        base_path = "../../../../data"
    else:
        base_path = "../../../../derived_data"

    return f"{base_path}/{horizontal_desc}/{rcp}/{ensemble_member}/{variable}/{temp_res}/{file_name}"

def load_dataset(horizontal_desc, source_model, variable, years, rcp="rcp85", ensemble_member="01", temp_res="day"):
    filepaths = [data_filepath(horizontal_desc, source_model, variable, year, rcp, ensemble_member, temp_res) for year in years]
    return xr.combine_by_coords([xr.load_dataset(filepath, decode_cf=True, decode_coords=True) for filepath in filepaths], combine_attrs="drop_conflicts", coords="all")
