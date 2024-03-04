# TO DO (after GH merge): python migrate-configs $DERIVED_DATA/moose/nc-datasets/*/ds-config.yml
# DONE: python migrate-configs src/mlde_data/configs/datasets/*.yml

import sys
import yaml

for fpath in sys.argv[1:]:
    print(fpath, flush=True)
    with open(fpath, "r") as f:
        config = yaml.safe_load(f)

    if "predictands" in config:
        print(f"Skipping {fpath}. Already in expected format.", flush=True)
        continue

    # backup existing config
    with open("{fpath}.bak", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    orig_predictand = config.pop("predictand")

    new_predictands = {
        "variables": [orig_predictand["variable"]],
        "resolution": orig_predictand["resolution"],
    }

    config["predictands"] = new_predictands

    orig_predictors = config.pop("predictors")
    orig_resolution = config.pop("resolution")

    new_predictors = {
        "variables": [v["variable"] for v in orig_predictors],
        "resolution": orig_resolution,
    }

    config["predictors"] = new_predictors

    if "split" in config:
        config["split"] = config.pop("split")

    print(f"Updating {fpath}.", flush=True)
    with open(fpath, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
