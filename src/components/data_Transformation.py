import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path :str  = os.path.join('artifacts','preprocessing.pkl')

class DataTransformation:
    def __init__(self):
        self.dataTranformaerConfig  = DataTransformationConfig()
    
    def get_data_tranformer_object(self):
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline  = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('Numerical Columns scaling is completed')
            cat_pipeline  = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ("oneHotEncoding",OneHotEncoder()),
                ]
            )
            logging.info('Categorical Columns encoding is completed')
            preprocessor  = ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_columns),
                    ('categorical_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
           train_df = pd.read_csv(train_path)
           test_df = pd.read_csv(test_path)  
           logging.info("Reading is completed for both test and train ")
           
           logging.info('Obtainin preprocessor object')
           preprocessing_obj = self.get_data_tranformer_object()
           
           target_column_name = 'math_score'
           numerical_columns = ['reading_score', 'writing_score']
           categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
           
           input_feature_train_df = train_df.drop(target_column_name,axis=1)
           target_feature_train_df  = train_df[target_column_name]
           
           input_feature_test_df = test_df.drop(target_column_name,axis=1)
           target_feature_test_df  = test_df[target_column_name]
           
           logging.info('applying preprocessing on test and training input featuers')
           input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
           input_feature_train_arr =preprocessing_obj.fit_transform(input_feature_train_df)
           
           train_arr = np.c_[input_feature_train_arr,np.array(input_feature_train_df)]
           test_arr = np.c_[input_feature_test_arr,np.array(input_feature_test_df)]
           logging.info('Data now is preprocessed')
           
           save_object(file_path = self.dataTranformaerConfig.preprocessor_obj_file_path,obj=preprocessing_obj)

           return (train_arr,test_arr,self.dataTranformaerConfig.preprocessor_obj_file_path)
           
        except Exception as e:
            raise CustomException(e,sys)