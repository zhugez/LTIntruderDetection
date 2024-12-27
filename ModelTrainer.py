from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelTrainer:
    """Handles model training and evaluation for intrusion detection."""

    def __init__(
        self,
        num_labels: int,
        model_name: str = "distilbert-base-uncased",
        batch_size: int = 32,
        max_length: int = 512,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        patience: int = 3,  # Early stopping patience
        min_delta: float = 1e-4,  # Minimum change threshold for early stopping
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.patience_counter = 3
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def prepare_data(
        self, texts: np.ndarray, labels: np.ndarray, val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training and validation."""

        # Tokenize texts
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create dataset
        dataset = TensorDataset(
            encodings["input_ids"], encodings["attention_mask"], torch.tensor(labels)
        )

        # Split dataset
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch and return average loss and accuracy."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            loss.backward()
            self.optimizer.step()

        return total_loss / len(train_loader), correct / total

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model and return average loss and accuracy."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )

                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(val_loader), correct / total

    def train(
        self, texts: np.ndarray, labels: np.ndarray, val_split: float = 0.2
    ) -> Dict[str, List[float]]:
        """Train the model with early stopping and return training history."""
        train_loader, val_loader = self.prepare_data(texts, labels, val_split)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        logger.info(f"Starting training on {self.device}")

        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            # Store metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
            )

            # Early stopping check
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_model("best_model")
                logger.info("New best model saved!")
            else:
                self.patience_counter += 1
                logger.info(
                    f"Early stopping counter: {self.patience_counter}/{self.patience}"
                )

                if self.patience_counter >= self.patience:
                    logger.info("Early stopping triggered!")
                    break

        # Load best model before returning
        self.load_model("best_model")
        self.plot_training_history()
        return self.history

    def plot_training_history(self, save_path: Optional[Path] = None) -> None:
        """Plot and save training history using plotly."""
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Model Loss", "Model Accuracy")
        )

        # Plot loss
        fig.add_trace(
            go.Scatter(y=self.history["train_loss"], name="Train Loss"), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=self.history["val_loss"], name="Validation Loss"), row=1, col=1
        )

        # Plot accuracy
        fig.add_trace(
            go.Scatter(y=self.history["train_acc"], name="Train Accuracy"), row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=self.history["val_acc"], name="Validation Accuracy"),
            row=1,
            col=2,
        )

        fig.update_layout(
            height=400, width=900, title_text="Training History", showlegend=True
        )

        # Save plot if path provided
        if save_path:
            fig.write_image(str(save_path))
            logger.info(f"Training history plot saved to {save_path}")
        else:
            fig.write_image("training_history.png")
            logger.info("Training history plot saved to training_history.png")

    def save_model(self, path: Path | str) -> None:
        """Save the model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "tokenizer")
        logger.info(f"Model and tokenizer saved to {path}")

    def load_model(self, path: Path | str) -> None:
        """Load a saved model and tokenizer."""
        path = Path(path)

        self.model = DistilBertForSequenceClassification.from_pretrained(path / "model")
        self.tokenizer = DistilBertTokenizer.from_pretrained(path / "tokenizer")
        self.model.to(self.device)
        logger.info(f"Model and tokenizer loaded from {path}")
