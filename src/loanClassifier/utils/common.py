import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from loanClassifier.custom_exception import CustomException
from loanClassifier.custom_logger import logger




def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logger.error(f'error saving object {e}')
        raise CustomException(e, sys)



def evaluate_model(X_train, y_train, X_test, y_test, models):

    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            # Make prediction
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            # precision = precision_score(y_test, predicted, average='weighted')
            # recall = recall_score(y_test, predicted, average='weighted')
            # f1 = f1_score(y_test, predicted, average='weighted')

            # {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1_score}

            report[list(models.keys())[i]] = accuracy

        return report

    except Exception as e:
        logger.error(f'error while evaluating {e}')
        raise CustomException(e, sys)
    
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logger.error(CustomException(e, sys))