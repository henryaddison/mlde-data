from string import Template
from pathlib import Path
import yaml


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
    return config
