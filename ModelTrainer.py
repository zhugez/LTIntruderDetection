from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from DataPreprocessor import CustomDistilBertTokenizer
from tqdm import tqdm
import os
import torch.distributed as dist
import time
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoConfig, AutoModelForSequenceClassification
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: CustomDistilBertTokenizer,
        batch_size: int = 32,
        max_length: int = 512,
        learning_rate: float = 5e-5,  # Increased base learning rate
        num_epochs: int = 50,
        patience: int = 7,  # Increased patience
        min_delta: float = 1e-4,
        use_flash_attention: bool = True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.patience_counter = 0  # Fix: Changed from 3 to 0
        self.use_flash_attention = use_flash_attention
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }
        self.scaler = torch.cuda.amp.GradScaler()
        self.optimizer = None
        self.scheduler = None  # Initialize these as None

    def prepare_data(
        self, texts: np.ndarray, labels: np.ndarray, val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for training and validation."""
        try:
            # Add data validation
            if len(texts) == 0 or len(labels) == 0:
                raise ValueError("Empty input data")

            # Log data statistics
            logger.info(f"Number of samples: {len(texts)}")
            logger.info(f"Number of unique labels: {len(np.unique(labels))}")
            logger.info(f"Label distribution: {np.bincount(labels)}")

            # Validate inputs
            if len(texts) != len(labels):
                raise ValueError(
                    f"Mismatched lengths: texts={len(texts)}, labels={len(labels)}"
                )

            logger.info(f"Processing {len(texts)} samples...")

            # Process texts in batches
            batch_size = min(
                1000, len(texts)
            )  # Ensure batch size isn't larger than dataset
            all_input_ids = []
            all_attention_masks = []

            for i in tqdm(range(0, len(texts), batch_size), desc="Processing texts"):
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts.tolist(),
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                if self.use_flash_attention:
                    encoded = {k: v.contiguous() for k, v in encoded.items()}

                all_input_ids.append(encoded["input_ids"])
                all_attention_masks.append(encoded["attention_mask"])

            # Concatenate and create tensors
            input_ids = torch.cat(all_input_ids, dim=0)
            attention_masks = torch.cat(all_attention_masks, dim=0)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

            # Create dataset
            dataset = TensorDataset(input_ids, attention_masks, labels_tensor)

            # Split dataset
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size

            if train_size == 0 or val_size == 0:
                raise ValueError(
                    f"Invalid split sizes: train={train_size}, val={val_size}"
                )

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=4 if torch.cuda.is_available() else 0,
                prefetch_factor=2 if torch.cuda.is_available() else None,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=2 if torch.cuda.is_available() else 0,
            )

            logger.info(
                f"Created dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches"
            )
            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            raise

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Verify optimizer state at start of epoch
        if self.optimizer is None:
            raise ValueError("Optimizer not initialized")

        init_lr = self.optimizer.param_groups[0]["lr"]
        logger.info(f"Starting epoch with learning rate: {init_lr}")

        for batch in tqdm(train_loader, desc="Training"):
            try:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    loss = outputs.loss

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update metrics
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Scheduler step (if using batch-level scheduling)
                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if current_lr == 0:
                        logger.warning("Learning rate dropped to 0!")

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                continue

        if total == 0:
            raise ValueError("No samples processed in training epoch")

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
        try:
            train_loader, val_loader = self.prepare_data(texts, labels, val_split)

            # Initialize optimizer if not already initialized
            if self.optimizer is None:
                # Modified optimizer configuration with higher initial learning rate
                param_optimizer = list(self.model.named_parameters())
                no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.01,
                        "lr": self.learning_rate,
                    },
                    {
                        "params": [
                            p
                            for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.learning_rate
                        * 10,  # Higher learning rate for bias and LayerNorm
                    },
                ]
                self.optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )

                # Log optimizer setup
                logger.info(f"Initialized optimizer with learning rates:")
                for idx, group in enumerate(self.optimizer.param_groups):
                    logger.info(f"Group {idx} learning rate: {group['lr']}")

            # Modified scheduler configuration
            if self.scheduler is None:
                num_training_steps = len(train_loader) * self.num_epochs
                num_warmup_steps = num_training_steps // 10  # 10% warmup steps

                # Use a more aggressive learning rate schedule
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps,
                    num_cycles=0.5,  # Add cycles for more aggressive schedule
                )
                logger.info(
                    f"Initialized scheduler with {num_warmup_steps} warmup steps"
                )

            # Add learning rate tracking
            self.history["learning_rates"] = []

            logger.info(f"Starting training on {self.device}")
            best_model_path = Path("best_model")

            # Modified early stopping criteria
            best_val_acc = 0.0
            patience_counter = 0

            for epoch in range(self.num_epochs):
                try:
                    train_loss, train_acc = self.train_epoch(train_loader)
                    val_loss, val_acc = self.evaluate(val_loader)

                    # Store metrics
                    self.history["train_loss"].append(train_loss)
                    self.history["train_acc"].append(train_acc)
                    self.history["val_loss"].append(val_loss)
                    self.history["val_acc"].append(val_acc)
                    self.history["learning_rates"].append(
                        self.optimizer.param_groups[0]["lr"]
                    )

                    logger.info(
                        f"Epoch {epoch+1}/{self.num_epochs} - "
                        f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                        f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )

                    # Modified early stopping based on accuracy
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        self.save_model(best_model_path)
                        logger.info(f"New best model saved! (Val Acc: {val_acc:.4f})")
                    else:
                        patience_counter += 1
                        logger.info(
                            f"Early stopping counter: {patience_counter}/{self.patience}"
                        )

                        if patience_counter >= self.patience:
                            logger.info("Early stopping triggered!")
                            break

                except Exception as e:
                    logger.error(f"Error in epoch {epoch+1}: {str(e)}")
                    continue

            # Load best model before returning
            if best_model_path.exists():
                self.load_model(best_model_path)

            self.plot_training_history()
            return self.history

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

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

    def save_model(self, path: Union[Path, str]) -> None:
        """Save the model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(path / "model")
        self.tokenizer.save_pretrained(path / "tokenizer")
        logger.info(f"Model and tokenizer saved to {path}")

    def load_model(self, path: Union[Path, str]) -> None:
        """Load a saved model and tokenizer."""
        path = Path(path)

        self.model = DistilBertForSequenceClassification.from_pretrained(path / "model")
        self.tokenizer = DistilBertTokenizer.from_pretrained(path / "tokenizer")
        self.model.to(self.device)
        logger.info(f"Model and tokenizer loaded from {path}")

    def _setup_model(self, num_labels: int) -> torch.nn.Module:
        """Setup model with optimized settings."""
        try:
            config = AutoConfig.from_pretrained(
                "distilbert-base-uncased",
                num_labels=num_labels,
                attention_probs_dropout_prob=0.1,
                hidden_dropout_prob=0.1,
                use_cache=False,
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                config=config,
            )

            if hasattr(model.config, "use_memory_efficient_attention"):
                model.config.use_memory_efficient_attention = True
                model.config.use_sdpa = True
                logger.info("Enabled memory efficient attention")

            return model

        except Exception as e:
            logger.error(f"Error setting up model: {str(e)}")
            raise

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "history": self.history,
        }
        filename = f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, filename)
        if is_best:
            shutil.copyfile(filename, "model_best.pt")
