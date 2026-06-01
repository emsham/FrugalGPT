import json
import os
import sys
from functools import lru_cache
from pathlib import Path


DEFAULT_FAKE_SERVICE_INFO = {
    "fake": {
        "support-cheap": {
            "cost_input": 0.00000005,
            "cost_output": 0.00000005,
            "cost_fixed": 0,
            "fixed_size": 0,
        },
        "support-strong": {
            "cost_input": 0.0000005,
            "cost_output": 0.0000005,
            "cost_fixed": 0,
            "fixed_size": 0,
        },
    }
}


def _candidate_config_dirs():
    env_dir = os.environ.get("FRUGALGPT_CONFIG_DIR")
    if env_dir:
        yield Path(env_dir)

    cwd = Path.cwd()
    yield cwd / "config"

    here = Path(__file__).resolve()
    for parent in here.parents:
        yield parent / "config"

    yield Path(sys.prefix) / "config"


def get_config_path(filename):
    for directory in _candidate_config_dirs():
        path = directory / filename
        if path.exists():
            return path
    searched = ", ".join(str(path / filename) for path in _candidate_config_dirs())
    raise FileNotFoundError(
        f"Could not find FrugalGPT config file {filename!r}. "
        f"Set FRUGALGPT_CONFIG_DIR or run from the repository root. Searched: {searched}"
    )


@lru_cache(maxsize=None)
def load_json_config(filename):
    with get_config_path(filename).open() as file:
        return json.load(file)


@lru_cache(maxsize=None)
def load_service_info():
    service_info = load_json_config("serviceinfo.json").copy()
    for provider, models in DEFAULT_FAKE_SERVICE_INFO.items():
        service_info.setdefault(provider, {}).update(models)
    return service_info
