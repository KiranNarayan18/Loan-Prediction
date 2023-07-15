import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

from loanClassifier.custom_exception import CustomException
from loanClassifier.custom_logger import logger
from loanClassifier.utils.common import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_obj(self):
        try:
            logger.info('Data Transformation Initiated')
            # Define which columns should be ordinal encoded and which should be scaled
            numerical_columns = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
            categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
            # Numerical Pipeline
            logger.info('Pipeline Initiated')
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())

                ]
            )


            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder', OneHotEncoder()),

                ]
            )


            prepreocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline, numerical_columns),
                ('cat_pipeline',  cat_pipeline, categorical_columns)
            ])

            logger.info('Pipeline Completed')

            return prepreocessor

        except Exception as e:
           logger.error(CustomException(e, sys))



    def initiate_data_transformation(self, train_data_path, test_data_path):
        """__Summary:"""

        logger.info("Data transformation initiated")

        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            #converting feature Dependents to int 
            train_df['Dependents'] = train_df['Dependents'].replace('3+', '3').replace('nan', np.nan)
            test_df['Dependents'] = test_df['Dependents'].replace('3+', '3').replace('nan', np.nan)
           
            #drop all null values
            train_df = train_df.dropna()
            test_df = test_df.dropna()

            train_df['Dependents'] = train_df['Dependents'].astype(int)
            test_df['Dependents'] = test_df['Dependents'].astype(int)

            #drop unnecessary features
            train_df = train_df.drop('Loan_ID',axis=1)
            test_df = test_df.drop('Loan_ID',axis=1)

            target_feature = 'Loan_Status'

            input_feature_train_df = train_df.drop(columns=target_feature, axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns=target_feature, axis=1)
            target_feature_test_df = test_df[target_feature]

            preprocessing_obj = self.get_data_transformation_obj()

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df)


            target_feature_train_df = target_feature_train_df.map({'N':0, 'Y':1})
            target_feature_test_df = target_feature_test_df.map({'N':0, 'Y':1})


            train_arr = np.c_[input_feature_train_arr,
                              np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,
                             np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logger.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            logger.error(CustomException(e, sys))



if __name__ == '__main__':
    obj = DataTransformation()
    train_arr,  test_arr, preprocessor_obj_file_path = obj.initiate_data_transformation('artifacts\\train.csv', 'artifacts\\test.csv')
    logger.info(f'result {preprocessor_obj_file_path}')