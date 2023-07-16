import sys

from loanClassifier.custom_exception import CustomException
from loanClassifier.custom_logger import logger
from loanClassifier.components.data_ingestion import DataIngestion
from loanClassifier.components.data_transformation import DataTransformation
from loanClassifier.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    try:
        
        data_ingestion_obj = DataIngestion()
        train_path, test_path = data_ingestion_obj.initiate_data_ingestion()

        data_transformation_obj = DataTransformation()
        train_arr, test_arr, preprocessor_obj_file_path = data_transformation_obj.initiate_data_transformation(train_path, test_path)

        model_trainer_obj = ModelTrainer()
        model_trainer_obj.initiate_model_training(train_arr, test_arr)
    except Exception as e:        
        logger.error(CustomException(e, sys))