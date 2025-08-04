# Model 2 NER Implementation

This implementation provides Model 2 from the NER_Prediction_Final.ipynb notebook, specifically designed to match the structure and parameters used in the original notebook.

## Overview

Model 2 is a Bidirectional LSTM-based Named Entity Recognition model that follows the exact architecture and parameters from the notebook:

- **Embedding Layer**: 50 dimensions
- **Bidirectional LSTM**: 100 units with 0.1 recurrent dropout
- **TimeDistributed Dense**: Softmax activation for tag classification
- **Optimizer**: Adam
- **Loss Function**: Categorical crossentropy
- **Training Parameters**: 10 epochs, batch size 64

## Key Features

### 1. Data Preprocessing
- Supports categorical (one-hot) encoding for tags as used in Model 2
- Maintains compatibility with sparse categorical encoding for other models
- Configurable maximum sequence length (default: 128, notebook uses 75)
- Handles vocabulary building with UNK and PAD tokens

### 2. Model Architecture
The `Model2NER` class implements the exact architecture from the notebook:

```python
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_len))
model.add(Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(num_tags, activation="softmax")))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
```

### 3. Training and Evaluation
- Supports the exact training parameters from the notebook
- Provides comprehensive training history tracking
- Includes plotting capabilities for loss and accuracy curves
- Model saving and loading functionality

## Usage

### Basic Usage

```python
from src.data_preprocessing import NERDataProcessor
from src.advanced_model import Model2NER, create_model2_ner

# Data preprocessing
processor = NERDataProcessor(max_sequence_length=75)
processed_data = processor.process_data("data/ner_dataset.csv", categorical_tags=True)

# Create Model 2
model = create_model2_ner(
    vocab_size=processed_data['metadata']['vocab_size'],
    num_tags=processed_data['metadata']['num_tags'],
    max_sequence_length=processed_data['metadata']['max_sequence_length']
)

# Train with notebook parameters
history = model.train(
    processed_data['X_train'],
    processed_data['y_train'],
    processed_data['X_val'],
    processed_data['y_val'],
    epochs=10,
    batch_size=64,
    verbose=1
)
```

### Running the Complete Pipeline

```bash
# Run the main demonstration
python main.py

# Run tests
python test_model2_simple.py    # Simple synthetic data test
python test_integration.py      # Complete pipeline test
```

## File Structure

```
src/
├── data_preprocessing.py    # Updated with categorical encoding support
├── advanced_model.py       # Contains Model2NER class
├── baseline_model.py       # Baseline model (unchanged)
├── evaluation.py           # Evaluation utilities
└── utils.py               # Utility functions

test_model2_simple.py       # Simple functionality test
test_integration.py         # Complete pipeline test
main.py                     # Full demonstration
```

## Model 2 vs Advanced Model

| Feature | Model 2 | Advanced Model |
|---------|---------|----------------|
| Architecture | Sequential BiLSTM | Multi-layer BiLSTM + Attention |
| Embedding Dim | 50 | 200 |
| LSTM Units | 100 | 128 (configurable) |
| Attention | None | Multi-head attention |
| Parameters | ~174K | >1M |
| Training Time | Fast | Slower |
| Complexity | Simple | Complex |

## Expected Performance

Based on the notebook results, Model 2 should achieve:
- Training accuracy: ~99.5%
- Validation accuracy: ~98.5%
- Training converges in ~10 epochs

## Backwards Compatibility

The implementation maintains backwards compatibility:
- Original `AdvancedNERModel` class is preserved
- Both sparse and categorical tag encoding supported
- Existing evaluation pipeline works with both models

## Dependencies

- TensorFlow/Keras >= 2.19.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.5.0 (for plotting)

## Testing

Run the test suite to verify functionality:

```bash
# Quick test with synthetic data
python test_model2_simple.py

# Complete integration test
python test_integration.py
```

Both tests should pass with output confirming:
- ✓ Model architecture matches notebook specifications
- ✓ Training works with categorical encoding
- ✓ Prediction and evaluation functions correctly
- ✓ Model saving/loading works

## Notes

1. **GPU Support**: The model will use GPU if available, but works fine on CPU
2. **Memory Usage**: Model 2 is lightweight and suitable for smaller datasets
3. **Extensibility**: The architecture can be easily modified while maintaining the core structure
4. **Notebook Compatibility**: This implementation produces results consistent with the original notebook

For questions or issues, refer to the test files for working examples.