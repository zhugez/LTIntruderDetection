import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Union
from dataclasses import dataclass
import torch.multiprocessing as mp
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from DataPreprocessor import DataPreprocessor, CustomDistilBertTokenizer
from ModelTrainer import ModelTrainer
import sys

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline."""

    # Data paths
    dataset_path: Union[str, Path] = "dataset.csv"
    model_output_path: Union[str, Path] = "saved_model"
    history_plot_path: Union[str, Path] = "training_history.png"

    # Training parameters
    batch_size: int = 32
    max_length: int = 512
    learning_rate: float = 1e-5
    num_epochs: int = 20
    validation_split: float = 0.2
    use_flash_attention: bool = True


class TrainingPipeline:
    """Handles the complete model training pipeline."""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.logger = self._setup_logging()

        # Initialize custom tokenizer
        self.tokenizer = CustomDistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", padding=True, truncation=True
        )

    def _setup_logging(self) -> logging.Logger:
        """Configure and return logger instance."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
        )
        return logging.getLogger(__name__)

    def _setup_model(self, num_labels: int) -> torch.nn.Module:
        """Setup model with optimized attention if available."""
        try:
            # Configure model with optimized settings
            config = AutoConfig.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_labels,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                use_cache=True,
            )

            # Use AutoModel for better compatibility
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                config=config,
                ignore_mismatched_sizes=True,  # Add this parameter
            )

            self.logger.info(
                "Note: Classifier layers are initialized randomly as expected"
            )

            if torch.cuda.is_available() and self.config.use_flash_attention:
                # Enable memory efficient attention if available
                if hasattr(model.config, "use_memory_efficient_attention"):
                    model.config.use_memory_efficient_attention = True
                    self.logger.info("Enabled memory efficient attention")
                else:
                    self.logger.info("Memory efficient attention not available")

            return model

        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            raise

    def train_model(self) -> Tuple[Dict[str, Any], ModelTrainer]:
        """Execute model training pipeline."""
        try:
            # Initialize preprocessor
            self.logger.info("Initializing data preprocessor...")
            preprocessor = DataPreprocessor(
                file_path=self.config.dataset_path, tokenizer=self.tokenizer
            )
            texts, labels = preprocessor.prepare_all_data()

            # Initialize model and trainer
            self.logger.info("Setting up model trainer...")
            num_labels = len(preprocessor.get_label_mapping())
            model = self._setup_model(num_labels)

            trainer = ModelTrainer(
                model=model,
                tokenizer=self.tokenizer,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                learning_rate=self.config.learning_rate,
                num_epochs=self.config.num_epochs,
                use_flash_attention=self.config.use_flash_attention,
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
            Path(self.config.model_output_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            Path(self.config.history_plot_path).parent.mkdir(
                parents=True, exist_ok=True
            )

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
    try:
        logger.info("Starting training pipeline...")

        # Check if dataset exists
        dataset_path = Path("dataset.csv")
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            return

        # Initialize config
        config = TrainingConfig(
            dataset_path=str(dataset_path),
            batch_size=32 * max(1, torch.cuda.device_count()),  # Reduced batch size
            num_epochs=50,  # Increased epochs
            learning_rate=2e-5,  # Adjusted learning rate
            use_flash_attention=True,
        )

        logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")

        if torch.cuda.device_count() > 1:
            logger.info("Using multi-GPU training...")
            mp.set_start_method("spawn", force=True)

            def train_on_device(rank: int):
                try:
                    logger.info(f"Initializing process on GPU {rank}")
                    torch.cuda.set_device(rank)
                    pipeline = TrainingPipeline(config)
                    pipeline.run()
                except Exception as e:
                    logger.error(f"Error in process {rank}: {str(e)}")
                    raise

            world_size = torch.cuda.device_count()
            mp.spawn(
                fn=train_on_device,
                args=(),
                nprocs=world_size,
            )
        else:
            logger.info("Using single device training...")
            pipeline = TrainingPipeline(config)
            pipeline.run()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
