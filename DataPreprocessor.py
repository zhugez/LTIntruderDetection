from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertTokenizer
from nltk.tokenize import RegexpTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CustomDistilBertTokenizer(DistilBertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Regex pattern to split and process tokens by ":"
        self.custom_tokenizer = RegexpTokenizer(r"[\w\-]+|[:=]|.+?(?=,|$)")

    def tokenize(self, text, **kwargs):
        """
        Tokenizes the input text using custom logic and splits by ":".
        """
        tokens = self.custom_tokenizer.tokenize(text)
        processed_tokens = [
            processed for token in tokens if (processed := self.process_token(token))
        ]
        return [
            item
            for sublist in processed_tokens
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

    def process_token(self, token):
        """
        Processes tokens and handles ":" as a separator.
        """
        if token == ":":
            return None  # Skip ":" but ensure it's recognized as a separator

        if ":" in token:
            key, value = token.split(":", 1)
            return [key.strip(), value.strip()]

        return [token]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts tokens to IDs using the parent's method.
        """
        return super().convert_tokens_to_ids(tokens)


@dataclass
class DataPreprocessor:
    """Handles data preprocessing for intrusion detection."""

    file_path: Path
    tokenizer: CustomDistilBertTokenizer
    label_encoder: LabelEncoder = LabelEncoder()
    data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load and validate data from CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            logger.info(f"Loaded data with columns: {list(self.data.columns)}")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _validate_data_loaded(self) -> None:
        """Validate that data is loaded before processing."""
        if self.data is None:
            self.load_data()

    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset by handling missing values and duplicates."""
        self._validate_data_loaded()
        self.data = self.data.fillna("")
        return self.data

    def combine_features(self) -> pd.DataFrame:
        """Combine and tokenize features into a single text field."""
        self._validate_data_loaded()

        def process_row(row: pd.Series) -> str:
            feature_pairs = [
                f"{col}: {str(val)}" for col, val in row.items() if col != "Class"
            ]
            text = ", ".join(feature_pairs)
            tokens = self.tokenizer.tokenize(text)
            return " ".join(tokens)

        self.data["Final"] = self.data.apply(process_row, axis=1)
        return self.data

    def encode_labels(self, label_column: str = "Class") -> pd.DataFrame:
        """Encode categorical labels while preserving original values."""
        self._validate_data_loaded()

        if label_column not in self.data.columns:
            raise ValueError(f"Label column '{label_column}' not found")

        self.data["Class_encoded"] = self.label_encoder.fit_transform(
            self.data[label_column]
        )
        logger.info(f"Encoded {len(self.label_encoder.classes_)} unique classes")
        return self.data

    def prepare_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Execute full data preparation pipeline."""
        self._validate_data_loaded()

        self.clean_data()
        self.combine_features()
        self.encode_labels()

        return self.data["Final"].values, self.data["Class_encoded"].values

    def get_label_mapping(self) -> Dict[int, Any]:
        """Get mapping between encoded and original labels."""
        if not hasattr(self.label_encoder, "classes_"):
            raise ValueError("Labels not yet encoded")

        return dict(enumerate(self.label_encoder.classes_))
