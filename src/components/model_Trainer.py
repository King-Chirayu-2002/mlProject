import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('Artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()  
    
    def initiateModelTrainer(self,train_arr,test_arr,preprocessor_path):
        try:
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
                )
            models = {
    'Linear Regression' : LinearRegression(),
    'K-Neighbours Regression' : KNeighborsRegressor(),
    'Gradient Boosting' : GradientBoostingRegressor(),
    'Decision Tree' : DecisionTreeRegressor(),
    'RandomForestRegressor' : RandomForestRegressor(),
    'AdaBoostRegression' : AdaBoostRegressor(),
    'CatBoost' : CatBoostRegressor(verbose=False),
    'XGBRegressor' : XGBRegressor()
}
            model_report : dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            else:
                logging.info('best model found on both data sets')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= models[best_model_name]
            )
        except Exception as e:
            raise CustomException(e,sys)
        