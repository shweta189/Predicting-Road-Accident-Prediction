import sys 
from src.entity.config_entity import IntroExtroPredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame

class PersonalityData:

    def __init__(self,
                 Time_spent_Alone,
                Stage_fear,
                Social_event_attendance,
                Going_outside,
                Drained_after_socializing,
                Friends_circle_size,
                Post_frequency,
                Personality):
        """
        Personality Data Constructor
        """
        try:
            self.Time_spent_Alone = Time_spent_Alone
            self.Stage_fear = Stage_fear
            self. Social_event_attendance = Social_event_attendance
            self.Going_outside = Going_outside
            self.Drained_after_socializing= Drained_after_socializing
            self.Friends_circle_size = Friends_circle_size
            self.Post_frequency = Post_frequency
            self.Personality = Personality
        except Exception as e:
            raise MyException(e,sys)
        
    def get_personality_input_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from PersonalityData class input
        """
        try:

            personality_input_dict = self.get_personality_input_as_dict()
            return DataFrame(personality_input_dict)
        except Exception as e:
            raise MyException(e,sys) from e 
        
    def get_personality_input_as_dict(self):
        """
        This function returns a dictionary from PersonalityData class input
        """
        logging.info("Entered get_personality_input_as_dict method as PersonalityData class")
        try:
            input_data= {
                "Time_spent_Alone": [self.Time_spent_Alone],
                "Stage_fear":[self.Stage_fear],
                "Social_event_attendance": [self.Social_event_attendance],
                "Going_outside":[self.Going_outside],
                "Drained_after_socializing":[self.Drained_after_socializing],
                "Friends_circle_size": [self.Friends_circle_size],
                "Post_frequency": [self.Post_frequency],
                "Personality": [self.Personality]

            }

            logging.info("Created personality data dict")
            logging.info("Exited get_personality_data_as_dict method as PersonalityData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e
        except Exception as e:
            raise MyException(e,sys) from e
        
class PersonalityDataClassifier:

    def __init__(self,prediction_pipeline_config:IntroExtroPredictorConfig = IntroExtroPredictorConfig(),) ->None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e,sys) from e
    
    def predict(self,dataframe) -> str:
        """
        This is the method of PersonalityDataClassifier
        Returns: Prediction is string format
        """
        try:
            logging.info("Entered predict method of PersonalityDataClassifier class")
            model = Proj1Estimator(
                bucket_name= self.prediction_pipeline_config.model_bucket_name,
                model_path= self.prediction_pipeline_config.model_file_path
            )
            result = model.predict(dataframe)
            print(result)
            return result
        except Exception as e:
            raise MyException(e,sys) from e