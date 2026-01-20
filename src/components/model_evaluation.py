from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact , DataIngestionArtifact,ModelEvaluationArtifact
from src.exception import MyException
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.constants import TARGET_COLUMN,SCHEMA_FILE_PATH
from src.logger import logging 
from src.utils.main_utils import load_object
import sys 
import pandas as pd 
from pandas import DataFrame
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from src.utils.main_utils import read_yaml_file
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float 
    best_model_f1_score: float 
    is_model_accepted: bool
    difference: float 

class ModelEvaluation:

    def __init__(self,model_eval_config:ModelEvaluationConfig,
                 data_ingestion_artifact:DataIngestionArtifact, model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path= SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys) from e 
        
    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name: get best model
        Description: This function is used to get get model from production stage

        Output : Returns model object if available in s3 storage
        On Failure : Write an exception log and then raise an exception
        """

        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None 
        except Exception as e:
            raise MyException(e,sys)
        
    def get_numerical_and_categorical_features(self) -> tuple:
        try:
            numerical_features = self._schema_config.get('num_features',[])
            categorical_features = self._schema_config.get('cat_features',[])

            if TARGET_COLUMN in categorical_features:
                categorical_features.remove(TARGET_COLUMN)
            logging.info(f"Numerical features : {numerical_features}")
            logging.info(f"categorical_features: {categorical_features}")

            return numerical_features, categorical_features
        except Exception as e:
            raise MyException(e,sys)
        
    # def handle_outliers(self,dataframe:DataFrame, num_features:list) -> DataFrame:
    #     try:
    #         logging.info("Handling outliers using IQR method")
    #         df = dataframe.copy()

    #         for col in num_features:
    #             if col in df.columns:
    #                 q1 = df[col].quantile(0.25)
    #                 q3 = df[col].quantile(0.75)
    #                 iqr = q3 -q1 
    #                 lower_bound = q1 - 1.5* iqr 
    #                 upper_bound = q3 + 1.5* iqr

    #                 df[col] = df[col].clip(lower= lower_bound,upper=upper_bound)
    #         logging.info(f"Outliers handled for numerical columns: {num_features}")
    #     except Exception as e:
    #         raise MyException(e,sys) from e
        
    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name: evaluate_model
        Description: This function is used to evalute trained model

        Output: Returns bool value based on validation result
        On Failure: Write an exception log and then raise an exception
        """
        try:
            column_names = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 
                           'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 
                           'Post_frequency', 'Personality']
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path,names=column_names,header = 0)
            x,y = test_df.drop(columns=[TARGET_COLUMN],axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded.")
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exist")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_score for this model: {trained_model_f1_score} ")

            best_model_f1_score = None
            best_model = self.get_best_model()

            if best_model is not None:
                logging.info(f"Computing F1_score for production model...")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y,y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New trained model: {trained_model_f1_score}")
            
            tmp_best_model_score =0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                             best_model_f1_score= best_model_f1_score,
                                             is_model_accepted= trained_model_f1_score> tmp_best_model_score,
                                              difference = trained_model_f1_score - tmp_best_model_score )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name: initiate_model_evaluation
        Description: This function is used to initiate all steps of model evaluation

        Output: Return model evaluation artifact
        On Failure: Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Intialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted= evaluate_model_response.is_model_accepted,
                s3_model_path= s3_model_path,
                trained_model_path= self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy= evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e,sys) from e