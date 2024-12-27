import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from dataclasses import dataclass
import torch.multiprocessing as mp
import torch
from DataPreprocessor import DataPreprocessor
from ModelTrainer import ModelTrainer


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline."""

    # Data paths
    dataset_path: str = "dataset.csv"
    model_output_path: str = "saved_model"
    history_plot_path: str = "training_history.png"

    # Training parameters
    batch_size: int = 32
    max_length: int = 512
    learning_rate: float = 2e-5
    num_epochs: int = 20
    validation_split: float = 0.2


class TrainingPipeline:
    """Handles the complete model training pipeline."""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Configure and return logger instance."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
        )
        return logging.getLogger(__name__)

    def train_model(self) -> Tuple[Dict[str, Any], ModelTrainer]:
        """Execute model training pipeline."""
        try:
            # Initialize preprocessor
            self.logger.info("Initializing data preprocessor...")
            preprocessor = DataPreprocessor(self.config.dataset_path)
            texts, labels = preprocessor.prepare_all_data()

            # Initialize trainer
            self.logger.info("Setting up model trainer...")
            num_labels = len(preprocessor.get_label_mapping())
            trainer = ModelTrainer(
                num_labels=num_labels,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                learning_rate=self.config.learning_rate,
                num_epochs=self.config.num_epochs,
            )

            # Train model
            self.logger.info("Starting model training...")
            history = trainer.train(
                texts, labels, val_split=self.config.validation_split
            )
            self.logger.info("Training completed successfully")

            return history, trainer

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def save_artifacts(self, trainer: ModelTrainer, history: Dict[str, Any]) -> None:
        """Save model and training artifacts."""
        try:
            # Create output directories if they don't exist
            Path(self.config.model_output_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            Path(self.config.history_plot_path).parent.mkdir(
                parents=True, exist_ok=True
            )

            # Save model and plot
            self.logger.info("Saving model artifacts...")
            trainer.save_model(self.config.model_output_path)
            trainer.plot_training_history(Path(self.config.history_plot_path))
            self.logger.info("Artifacts saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save artifacts: {str(e)}")
            raise

    def run(self) -> None:
        """Execute the complete training pipeline."""
        self.logger.info("Starting training pipeline")

        try:
            history, trainer = self.train_model()
            self.save_artifacts(trainer, history)
            self.logger.info("Training pipeline completed successfully")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main execution function with multi-GPU support."""
    config = TrainingConfig(
        dataset_path="dataset.csv",
        batch_size=64 * max(1, torch.cuda.device_count()),  # Scale batch size with GPUs
        num_epochs=30,
    )

    if torch.cuda.device_count() > 1:
        mp.spawn(
            TrainingPipeline(config).run, nprocs=torch.cuda.device_count(), join=True
        )
    else:
        pipeline = TrainingPipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()
