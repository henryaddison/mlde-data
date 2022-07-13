import iris

gcm_target_grid_pp_path = "pp-data/*.pp"
gcm_target_grid_with_bnds_path = "moose_grid.nc"
target_lat_name = 'latitude'
target_lon_name = 'longitude'

target_cube = iris.load_cube(gcm_target_grid_pp_path)

target_cube.coord(target_lon_name).guess_bounds()
target_cube.coord(target_lat_name).guess_bounds()

iris.save(target_cube, gcm_target_grid_with_bnds_path)
