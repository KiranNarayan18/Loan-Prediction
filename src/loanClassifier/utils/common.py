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



def evaluate_model123(X_train, y_train, X_test, y_test, models):
    try:

        model_list = list()
        accuracy_list = list()

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1 = f1_score(y_test, predicted, average='weighted')
           

            print(list(models.keys())[i])
            model_list.append(model)


            # print('----------------------------------')
            
            print('Model performance for Testing set')
            print("- accuracy: {:.4f}".format(accuracy_test))
            print("- precision: {:.4f}".format(precision_test))
            print("- recall: {:.4f}".format(recall_test))
            print("- f1: {:.4f}".format(f1_test))

            accuracy_list.append(f1_test)




    except Exception as e:
        logger.error(f'error while evaluating {e}')
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