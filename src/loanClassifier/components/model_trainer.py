import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


from loanClassifier.custom_exception import CustomException
from loanClassifier.custom_logger import logger
from loanClassifier.utils.common import save_object, evaluate_model
from loanClassifier.components.data_transformation import DataTransformation

@dataclass
class ModelTraningConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTraningConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logger.info(
                'splitting dependent and independent columns from train and test array')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]

            )

            models = {
                    "Logistic Regression": LogisticRegression(),
                    "Naive Bayes": GaussianNB(),
                    "K-Neighbors Classifier": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                    "support vector machine": SVC()
                }

            model_report: dict = evaluate_model(
                X_train, y_train, X_test, y_test, models)

            
            logger.info('\n====================================================================================\n')
            logger.info(f'Model Report : {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            # print('\n====================================================================================\n')
            logger.info(
                f'Best Model Found , Model Name : {best_model_name} , accuracy : {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logger.info('error occured at model training ')
            raise CustomException(e, sys)


if __name__ == '__main__':

    obj = DataTransformation()
    train_arr,  test_arr, preprocessor_obj_file_path = obj.initiate_data_transformation('artifacts\\train.csv', 'artifacts\\test.csv')
    logger.info(f'result {preprocessor_obj_file_path}')

    obj = ModelTrainer()
    obj.initiate_model_training(train_arr, test_arr)