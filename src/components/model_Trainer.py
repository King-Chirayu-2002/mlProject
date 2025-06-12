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
            from sklearn.model_selection import GridSearchCV

            param_grid = {
    'Linear Regression': {
        'fit_intercept': [True, False],
        'positive': [True, False]
    },

    'K-Neighbours Regression': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    },

    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },

    'Decision Tree': {
        'criterion': ['squared_error', 'friedman_mse'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },

    'RandomForestRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2', None]  # ✅ FIXED: removed 'auto', added None
    },

    'AdaBoostRegression': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 1.0],
        'loss': ['linear', 'square', 'exponential']
    },

    'CatBoost': {
        'iterations': [100, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    },

    'XGBRegressor': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }
}

            model_report : dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models,params = param_grid)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            else:
                logging.info('best model found on both data sets')
            
            print(model_report)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= models[best_model_name]
            )
        except Exception as e:
            raise CustomException(e,sys)
        