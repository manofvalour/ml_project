import os
import sys
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

from src.exception import CustomException
from src.utils import load_obj



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path= 'artifact/preprocessor.pkl'

            model= load_obj(model_path)
            preprocess= load_obj(preprocessor_path)

            preprocessed= preprocess.transform(features)
            pred= model.predict(features)

            return pred
        
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self, gender:str, age:int,
                 race_ethnicity:str, parental_level_of_education: str,
                lunch:str, test_preparation_course:str,
                reading_score:int,
                writing_score:int):
        
        self.gender= gender
        self.age=age
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education= parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    
    def get_data_to_dataframe(self):
        try:
            custom_data_input_dict={
                'gender':self.gender,
                'age': self.age,
                'race_ethnicity': self.race_ethnicity,
                'parental_level_of_education': self.parental_level_of_education,
                'lunch': self.lunch,
                'test_preparation_course':self.test_preparation_course,
                'reading_score': self.reading_score,
                'writing_score': self.writing_score
            }

            dataframe=pd.DataFrame(custom_data_input_dict)

            return dataframe
        
        except Exception as e:
            raise CustomException(e, sys)