import numpy as np
from config.python_config import config
from pipeline import chocolate_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


# train model 
def run_training() -> None:
    """Train the model."""

    # read training data
    #----READ THE DATASET DEFINED IN THE CONFIG.YAML FILE ----# 
    data = load_dataset(file_name=config.training_data_file)



    # divide train and test
    # SPLIT INTO TRAIN, TEST WITH THE DATA FED 
    X_train, X_test, y_train, y_test = train_test_split(
        
        data[config.model_config.features],  # predictors (we keep all the features since the pipeline includes feature selection)
        data[config.model_config.target],    # select target 
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    

    # fit model
    chocolate_pipe.fit(X_train, y_train)

    # persist trained model
    save_pipeline(pipeline_to_persist=chocolate_pipe)


if __name__ == "__main__":
    run_training()

