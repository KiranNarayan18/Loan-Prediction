import os
import sys
import pandas as pd

from loanClassifier.custom_exception import CustomException
from loanClassifier.custom_logger import logger
from loanClassifier.utils.common import load_object


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logger.error(CustomException(e, sys))
            raise CustomException(e, sys)


class CustomData:
    
    def __init__(self, Gender: str, Married: str, Dependents: int, Education: str, Self_Employed: str, ApplicantIncome: float, CoapplicantIncome: float, LoanAmount: float, Loan_Amount_Term: int, Credit_History: float, Property_Area: str):

        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Gender': [self.Gender],
                'Married': [self.Married],
                'Dependents': [self.Dependents],
                'Education': [self.Education],
                'Self_Employed': [self.Self_Employed],
                'ApplicantIncome': [self.ApplicantIncome],
                'CoapplicantIncome': [self.CoapplicantIncome],
                'LoanAmount': [self.LoanAmount],
                'Loan_Amount_Term': [self.Loan_Amount_Term],
                'Loan_Amount_Term': [self.Credit_History],
                'Loan_Amount_Term': [self.Property_Area]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logger.info('Dataframe Gathered')
            return df
        except Exception as e:
            logger.error(CustomException(e, sys))
            raise CustomException(e, sys)
        


