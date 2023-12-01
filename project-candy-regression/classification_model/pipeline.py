import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso,Ridge
from sklearn.metrics import mean_squared_error,r2_score,confusion_matrix,ConfusionMatrixDisplay,roc_curve,RocCurveDisplay,auc,precision_recall_fscore_support,precision_score
import joblib 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,Binarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from feature_engine.wrappers import SklearnTransformerWrapper as TransformerWrapper
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from classification_model.config.python_config import config


# continuous variables
chocolate_pipe = Pipeline([   
    ('categoricalimputer',CategoricalImputer(imputation_method='frequent' ,variables=config.model_config.discreet_na,ignore_format=True)),
    ('medianimputer',MeanMedianImputer(imputation_method='median',variables=config.model_config.continuous_na)),
    ('scaler', MinMaxScaler()),
    ( 'featuresel', SelectFromModel (estimator=LinearSVC(C=config.model_config.c,verbose=3,random_state=config.model_config.random_state),max_features=config.model_config.num_features)) ,
    ('model',LinearSVC(C=config.model_config.c,verbose=3,random_state=config.model_config.random_state) )
])


