from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError
from classification_model.config.python_config import config



def validate_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame :
    """Check model inputs for unprocessable values."""

    # check that we have read the appropriate features from the config.yaml file 
    # HERE WE SELECT ONLY THE FEATURES FROM THE DATASET 
    relevant_data = input_data[config.model_config.features].copy()
    errors = None

    try:
    
        CandyInputs(
            inputs=relevant_data
        )

        
    except ValidationError as error:
        errors = error.json()

    return relevant_data, errors



#  set the schema for the features of the dataset 
class CandySchema(BaseModel):
  sugarpercent : float
  pricepercent : float 
  winpercent : float 
  fruity  : int 
  caramel : int 
  peanutyalmondy : int
  nougat : int
  crispedricewafer : int
  hard : int 
  bar : int 
  pluribus : int 
 
# inputs are a list of the features 
class CandyInputs(BaseModel):
    inputs: List[CandySchema]
