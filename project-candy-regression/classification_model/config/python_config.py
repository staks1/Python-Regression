from pydantic import BaseModel
from strictyaml import YAML,load 
from pathlib import Path 
from typing import Dict, List, Optional, Sequence
import classification_model


# we import typing , to use predefined types for static type checking 
# pydantic is used for schema and type validation 

# set up environmental variables 
PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH : PACKAGE_ROOT / "config.yaml"
DATASET_DIR = PACKAGE_ROOT / "Datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "Models-Transformers"
MODEL_NAME = 'chocolate-candy-classifier'

# set up the schema and functions 

class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
    # general features and target to predict 
    alpha : float 
    c : float 
    target: str
    features: List[str]
    test_size: float
    random_state: int
    # we need  also the discreet and continuous variables that we impute 
    discreet_na : List[str]
    continuous_na : List[str]
    num_features : int = len(features)


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()