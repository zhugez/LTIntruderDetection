# DistilBERT Training Pipeline for Intrusion Detection

A comprehensive pipeline for training DistilBERT models on network intrusion detection data, featuring advanced data preprocessing, custom tokenization, and optimized model architecture.

## 📊 Data Processing Pipeline

### Custom Tokenization Strategy

- **Domain-Specific Tokenizer**
  - Extended DistilBERT tokenizer with network protocol awareness
  - Special handling for key-value pairs (e.g., "protocol:TCP")
  - Custom regex patterns for network traffic features
  ```python
  custom_tokenizer = RegexpTokenizer(r"[\w\-]+|[:=]|.+?(?=,|$)")
  ```

### Data Preprocessing Steps

1. **Data Loading & Validation**

   - Automatic CSV format detection
   - Column type validation
   - Missing value detection
   - Data integrity checks

2. **Feature Engineering**

   - Intelligent feature combination
   - Key-value pair formatting
   - Protocol-specific processing
   - Sequence length optimization

3. **Label Processing**
   - Label encoding with mapping preservation
   - Class distribution analysis
   - Imbalance detection
   - Multi-class support

## 🧠 Model Architecture

### Base Model Configuration

- **DistilBERT Base**
  - 6 transformer layers
  - 768 hidden dimensions
  - 12 attention heads
  - 66M parameters

### Optimizations

1. **Attention Mechanism**

   - Flash Attention 2.0 integration
   - Memory-efficient attention
   - Scaled dot-product attention
   - Custom attention patterns

2. **Training Enhancements**
   - Mixed precision training (FP16)
   - Gradient accumulation
   - Gradient checkpointing
   - Memory optimization

### Learning Rate Strategy

- **Dual Learning Rate System**
  ```python
  optimizer_groups = [
      {
          "params": backbone_params,
          "lr": base_lr,
          "weight_decay": 0.01
      },
      {
          "params": classifier_params,
          "lr": base_lr * 10,
          "weight_decay": 0.0
      }
  ]
  ```

## 📈 Training Process

### Initialization

1. **Model Setup**

   - Custom configuration loading
   - Weight initialization
   - GPU memory optimization
   - Multi-GPU preparation

2. **Data Pipeline**
   - DataLoader configuration
   - Prefetch buffer setup
   - Memory pinning
   - Batch size optimization

### Training Loop

1. **Forward Pass**

   - Automatic mixed precision
   - Gradient scaling
   - Loss computation
   - Metric tracking

2. **Backward Pass**
   - Gradient clipping
   - Optimizer stepping
   - Learning rate scheduling
   - State updates

### Monitoring

- **Real-time Metrics**
  - Loss tracking
  - Accuracy monitoring
  - Learning rate progression
  - Resource utilization

## 🔍 Performance Analysis

### Metrics

- Training/Validation Loss
- Accuracy per class
- F1 Score
- Confusion Matrix

### Visualization

- Training history plots
- Learning rate curves
- Resource utilization
- Model performance analysis
