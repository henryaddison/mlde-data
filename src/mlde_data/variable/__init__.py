from dataclasses import dataclass
from pathlib import Path
from string import Template
import yaml

from mlde_data.options import CollectionOption


@dataclass(frozen=True)
class SourceVariableConfig:
    src_type: str
    collection: str
    frequency: str
    variable: str
    resolution: str = None
    domain: str = None

    def __post_init__(self):
        if self.src_type == "moose":
            if self.collection == CollectionOption.cpm:
                if self.resolution is None:
                    object.__setattr__(self, "resolution", "2.2km")
                if self.domain is None:
                    object.__setattr__(self, "domain", "uk")
            elif self.collection == CollectionOption.gcm:
                if self.resolution is None:
                    object.__setattr__(self, "resolution", "60km")
                if self.domain is None:
                    object.__setattr__(self, "domain", "global")
            else:
                raise f"Unknown collection {self.collection}"
        elif self.src_type == "local":
            # assume local sourced data is pre-processed so resolution and domain must be specified
            assert (
                self.resolution is not None
            ), "resolution must be specified for local source variable"
            assert (
                self.domain is not None
            ), "domain must be specified for local source variable"
        elif self.src_type == "canari-le-sprint":
            # assume CANARI LE Sprint data is at global 60km resolution
            if self.resolution is None:
                object.__setattr__(self, "resolution", "60km")
            if self.domain is None:
                object.__setattr__(self, "domain", "global")
        else:
            raise RuntimeError(f"Unknown souce type {self.src_type}")


def load_config(
    config_path: Path,
    scale_factor: str,
    domain: str,
    theta: int = None,
    target_resolution: str = None,
):
    """Load configuration for a creating a variable from a YAML file."""

    with open(config_path, "r") as config_template:
        d = {
            "scale_factor": scale_factor,
            "domain": domain,
            "theta": theta,
            "target_resolution": target_resolution,
        }
        src = Template(config_template.read())
        result = src.substitute(d)
        config = yaml.safe_load(result)

    config["sources"] = {
        SourceVariableConfig(
            src_type=config["sources"]["type"],
            collection=config["sources"]["collection"],
            frequency=config["sources"]["frequency"],
            variable=var_configs["name"],
        )
        for var_configs in config["sources"]["variables"]
    }

    return config
