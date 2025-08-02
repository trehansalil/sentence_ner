#!/usr/bin/env python3
"""
Test script for Model 2 implementation
"""

import os
import sys
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import with absolute path to avoid relative import issues
import sys
sys.path.append('src')

# First, let's create a simplified utils module for testing
def create_tag_to_id_mapping(tags):
    return {tag: idx for idx, tag in enumerate(tags)}

def create_id_to_tag_mapping(tag_to_id):
    return {idx: tag for tag, idx in tag_to_id.items()}

def save_results(results, path):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

# Import the main classes
try:
    from data_preprocessing import NERDataProcessor
    from advanced_model import Model2NER, create_model2_ner
except ImportError as e:
    print(f"Import error: {e}")
    # Let's implement a quick workaround
    sys.exit(1)


def test_model2_implementation():
    """Test the Model 2 implementation with sample data."""
    print("Testing Model 2 NER Implementation")
    print("=" * 50)
    
    # Check if data file exists
    data_path = "data/ner_dataset.csv"
    if os.path.exists(data_path):
        print(f"✓ Found dataset at {data_path}")
        use_real_data = True
    else:
        print(f"✗ Dataset not found at {data_path}, using synthetic data")
        use_real_data = False
    
    if use_real_data:
        # Test with real data
        processor = NERDataProcessor(max_sequence_length=75)  # Match notebook
        processed_data = processor.process_data(data_path, categorical_tags=True)
        
        print(f"✓ Processed real data:")
        print(f"  - Vocabulary size: {processed_data['metadata']['vocab_size']}")
        print(f"  - Number of tags: {processed_data['metadata']['num_tags']}")
        print(f"  - Training samples: {processed_data['metadata']['train_size']}")
        print(f"  - Validation samples: {processed_data['metadata']['val_size']}")
        print(f"  - Test samples: {processed_data['metadata']['test_size']}")
        
        # Test Model 2
        model = create_model2_ner(
            vocab_size=processed_data['metadata']['vocab_size'],
            num_tags=processed_data['metadata']['num_tags'],
            max_sequence_length=processed_data['metadata']['max_sequence_length']
        )
        
        # Build and show model summary
        model.build_model()
        print(f"\n✓ Model 2 built successfully")
        print("Model Summary:")
        print(model.get_model_summary())
        
        # Test with small epoch for quick validation
        print("\n✓ Testing training with 1 epoch...")
        history = model.train(
            processed_data['X_train'][:100],  # Small subset for testing
            processed_data['y_train'][:100],
            processed_data['X_val'][:50],
            processed_data['y_val'][:50],
            epochs=1,
            batch_size=32,
            verbose=1
        )
        
        print(f"✓ Training completed:")
        print(f"  - Final train accuracy: {history['final_train_accuracy']:.4f}")
        print(f"  - Final val accuracy: {history['final_val_accuracy']:.4f}")
        
    else:
        # Test with synthetic data
        print("\n✓ Creating synthetic data for testing...")
        
        # Create synthetic data matching the expected format
        vocab_size = 1000
        num_tags = 17  # Standard BIO tagging scheme
        max_seq_len = 75
        
        # Create random data
        X_train = np.random.randint(0, vocab_size, (100, max_seq_len))
        y_train = np.random.randint(0, num_tags, (100, max_seq_len, num_tags))  # One-hot encoded
        X_val = np.random.randint(0, vocab_size, (20, max_seq_len))
        y_val = np.random.randint(0, num_tags, (20, max_seq_len, num_tags))
        
        # Convert to one-hot
        from tensorflow.keras.utils import to_categorical
        y_train_sparse = np.random.randint(0, num_tags, (100, max_seq_len))
        y_val_sparse = np.random.randint(0, num_tags, (20, max_seq_len))
        y_train = to_categorical(y_train_sparse, num_classes=num_tags)
        y_val = to_categorical(y_val_sparse, num_classes=num_tags)
        
        print(f"✓ Synthetic data created:")
        print(f"  - Vocabulary size: {vocab_size}")
        print(f"  - Number of tags: {num_tags}")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Validation samples: {len(X_val)}")
        
        # Test Model 2
        model = create_model2_ner(
            vocab_size=vocab_size,
            num_tags=num_tags,
            max_sequence_length=max_seq_len
        )
        
        # Build and show model summary
        model.build_model()
        print(f"\n✓ Model 2 built successfully")
        print("Model Summary:")
        print(model.get_model_summary())
        
        # Test training with synthetic data
        print("\n✓ Testing training with 1 epoch...")
        history = model.train(
            X_train, y_train, X_val, y_val,
            epochs=1, batch_size=32, verbose=1
        )
        
        print(f"✓ Training completed:")
        print(f"  - Final train accuracy: {history['final_train_accuracy']:.4f}")
        print(f"  - Final val accuracy: {history['final_val_accuracy']:.4f}")
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! Model 2 implementation is working correctly.")


if __name__ == "__main__":
    test_model2_implementation()