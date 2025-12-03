import os
import sys
import json
from dataclasses import dataclass

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object

from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer


@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = os.path.join("artifacts")
    metrics_file: str = os.path.join("artifacts", "metrics.json")


class TrainingPipeline:
    def __init__(self, config: TrainingPipelineConfig = TrainingPipelineConfig()):
        self.config = config
        os.makedirs(self.config.artifacts_dir, exist_ok=True)
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")

            # 1. Ingest data (produces train/test csv paths)
            train_path, test_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. train: {train_path}, test: {test_path}")

            # 2. Data transformation (returns transformed arrays and preprocessor path)
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_path, test_path
            )
            logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")

            # 3. Model training
            r2_score_value = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed. r2_score: {r2_score_value}")

            # 4. Persist metrics
            metrics = {"r2_score": float(r2_score_value)}
            with open(self.config.metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            logging.info(f"Metrics saved to {self.config.metrics_file}")

            return {
                "train_path": train_path,
                "test_path": test_path,
                "preprocessor_path": preprocessor_path,
                "metrics_path": self.config.metrics_file,
                "metrics": metrics,
            }

        except Exception as e:
            raise CustomException(e, sys)
