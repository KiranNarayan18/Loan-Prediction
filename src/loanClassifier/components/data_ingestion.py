import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from loanClassifier.custom_exception import CustomException
from loanClassifier.custom_logger import logger


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """_summary_

        """
        logger.info('Data Ingestion stage started')
        try:
            df = pd.read_csv(os.path.join('notebooks','data', 'loanPrediction.csv'))

            logger.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(    
                self.ingestion_config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False)


            train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header= True)

            logger.info('train test split completed')


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            logger.error(CustomException(e, sys))



if __name__ == '__main__':
    obj = DataIngestion()
    result = obj.initiate_data_ingestion()
    print(result)