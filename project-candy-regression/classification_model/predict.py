import typing as t
import numpy as np
import pandas as pd
from classification_model import __version__ as _version
from classification_model.config.python_config import config
from classification_model.processing.data_manager import load_pipeline
from classification_model.processing.validation import validate_inputs


# read pipeline name 
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"

# we load the saved pipeline to make predictions 
_chocolate_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict ],) -> dict:
    """Make a prediction using a saved model pipeline."""

    #read Dataframe of features 
    data = pd.DataFrame(input_data)

    # read the data 
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    # check for errors in the data 
    if not errors:
        predictions = _chocolate_pipe.predict(
            X=validated_data[config.model_config.features]
        )

        # make the prediction 
        results = {
            "predictions": [pred for pred in predictions],  # type: ignore
            "version": _version,
            "errors": errors,
        }

    return results
