import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
LEGACY_PROJECTS_DIR = PROJECT_ROOT.parent


def _load_env_file() -> None:
    """Read KEY=VALUE lines of the local .env file, set variables take precedence."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


def resolve_config_path(config_name: str) -> Path:
    return (CONFIG_DIR / config_name).resolve()


def resolve_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR.resolve()


def resolve_lut_path() -> Path:
    configured = os.getenv("MODELICA_LUT_PATH")
    if configured:
        return Path(configured).expanduser()
    return resolve_results_dir() / "OpCons_data.txt"


def resolve_measurement_csv() -> Path | None:
    """Return the measurement file that replaces the database, if FCS_MEASUREMENT_CSV is set.

    The published record provides the raw file of the campaign for this purpose.
    """
    configured = os.getenv("FCS_MEASUREMENT_CSV")
    return Path(configured).expanduser() if configured else None


def resolve_simulation_root() -> Path:
    configured = os.getenv("FCS_SIMULATION_ROOT")
    if configured:
        return Path(configured).expanduser()
    return resolve_results_dir() / "simulation"


def resolve_download_dir() -> Path:
    configured = os.getenv("FCS_DOWNLOAD_DIR")
    if configured:
        return Path(configured).expanduser()
    download_dir = resolve_results_dir() / "download"
    download_dir.mkdir(exist_ok=True)
    return download_dir


def map_nested_sim_config(config_path: str | Path) -> Path:
    return (CONFIG_DIR / Path(config_path).name).resolve()
