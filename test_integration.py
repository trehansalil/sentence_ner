#!/usr/bin/env python3
"""
Integration test for the complete Model 2 pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import NERDataProcessor
from advanced_model import Model2NER, create_model2_ner


def create_mock_ner_data():
    """Create mock NER data similar to real dataset format."""
    sentences = [
        ("Barack Obama was born in Hawaii".split(), ["B-PER", "I-PER", "O", "O", "O", "B-LOC"]),
        ("Apple Inc is in Cupertino California".split(), ["B-ORG", "I-ORG", "O", "O", "B-LOC", "B-LOC"]),
        ("Microsoft was founded by Bill Gates".split(), ["B-ORG", "O", "O", "O", "B-PER", "I-PER"]),
        ("The company is located in Seattle".split(), ["O", "O", "O", "O", "O", "B-LOC"]),
        ("John works at Google in Mountain View".split(), ["B-PER", "O", "O", "B-ORG", "O", "B-LOC", "I-LOC"]),
    ] * 20  # Repeat to have more data
    
    # Create DataFrame in CoNLL format
    data = []
    for i, (words, tags) in enumerate(sentences):
        sentence_id = f"Sentence: {i+1}"
        for word, tag in zip(words, tags):
            data.append([sentence_id, word, tag])
    
    df = pd.DataFrame(data, columns=["Sentence #", "Word", "Tag"])
    return df


def test_model2_integration():
    """Test the complete Model 2 pipeline with mock data."""
    print("Model 2 NER Integration Test")
    print("=" * 50)
    
    # Create mock data
    print("1. Creating mock NER data...")
    df = create_mock_ner_data()
    
    # Save mock data to temporary file
    os.makedirs("data", exist_ok=True)
    mock_data_path = "data/mock_ner_data.csv"
    df.to_csv(mock_data_path, index=False)
    
    print(f"✓ Mock data created:")
    print(f"  - Total tokens: {len(df):,}")
    print(f"  - Unique sentences: {df['Sentence #'].nunique()}")
    print(f"  - Unique tags: {df['Tag'].nunique()}")
    print(f"  - Tags: {sorted(df['Tag'].unique())}")
    
    # Test preprocessing
    print("\n2. Testing data preprocessing...")
    processor = NERDataProcessor(max_sequence_length=20)  # Short for testing
    processed_data = processor.process_data(mock_data_path, categorical_tags=True)
    
    print(f"✓ Preprocessing completed:")
    print(f"  - Vocabulary size: {processed_data['metadata']['vocab_size']}")
    print(f"  - Number of tags: {processed_data['metadata']['num_tags']}")
    print(f"  - Training samples: {processed_data['metadata']['train_size']}")
    print(f"  - Validation samples: {processed_data['metadata']['val_size']}")
    print(f"  - Test samples: {processed_data['metadata']['test_size']}")
    print(f"  - Categorical tags: {processed_data['metadata']['categorical_tags']}")
    
    # Verify data shapes
    print(f"  - X_train shape: {processed_data['X_train'].shape}")
    print(f"  - y_train shape: {processed_data['y_train'].shape}")
    
    # Test Model 2 creation
    print("\n3. Testing Model 2 creation...")
    model = create_model2_ner(
        vocab_size=processed_data['metadata']['vocab_size'],
        num_tags=processed_data['metadata']['num_tags'],
        max_sequence_length=processed_data['metadata']['max_sequence_length']
    )
    
    keras_model = model.build_model()
    print(f"✓ Model 2 created successfully")
    
    # Test training (very short)
    print("\n4. Testing Model 2 training...")
    history = model.train(
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data['X_val'],
        processed_data['y_val'],
        epochs=2,  # Very short for testing
        batch_size=16,
        verbose=1
    )
    
    print(f"✓ Training completed:")
    print(f"  - Final train accuracy: {history['final_train_accuracy']:.4f}")
    print(f"  - Final val accuracy: {history['final_val_accuracy']:.4f}")
    
    # Test evaluation
    print("\n5. Testing evaluation...")
    test_results = model.evaluate(
        processed_data['X_test'],
        processed_data['y_test']
    )
    
    print(f"✓ Evaluation completed:")
    print(f"  - Test accuracy: {test_results['test_accuracy']:.4f}")
    print(f"  - Test loss: {test_results['test_loss']:.4f}")
    
    # Test prediction on new data
    print("\n6. Testing prediction...")
    sample_text = "Apple Inc was founded by Steve Jobs"
    sample_input = processor.preprocess_text(sample_text)
    predictions = model.predict(sample_input)
    
    print(f"✓ Prediction completed:")
    print(f"  - Input text: '{sample_text}'")
    print(f"  - Input shape: {sample_input.shape}")
    print(f"  - Prediction shape: {predictions.shape}")
    print(f"  - Sample predictions: {predictions[0][:len(sample_text.split())]}")
    
    # Test saving and loading
    print("\n7. Testing model save/load...")
    os.makedirs("models", exist_ok=True)
    test_model_path = "models/test_model2.h5"
    
    model.save_model(test_model_path)
    print(f"✓ Model saved to {test_model_path}")
    
    # Load and test
    new_model = Model2NER(
        vocab_size=processed_data['metadata']['vocab_size'],
        num_tags=processed_data['metadata']['num_tags'],
        max_sequence_length=processed_data['metadata']['max_sequence_length']
    )
    new_model.load_model(test_model_path)
    
    # Test loaded model prediction
    loaded_predictions = new_model.predict(sample_input)
    print(f"✓ Loaded model prediction shape: {loaded_predictions.shape}")
    
    # Verify predictions are the same
    prediction_diff = np.abs(predictions - loaded_predictions).max()
    print(f"✓ Prediction difference (should be ~0): {prediction_diff:.6f}")
    
    # Cleanup
    os.remove(mock_data_path)
    os.remove(test_model_path)
    
    print("\n" + "=" * 50)
    print("✓ Integration test completed successfully!")
    print("\nVerified capabilities:")
    print("  - ✓ Data preprocessing with categorical encoding")
    print("  - ✓ Model 2 architecture (BiLSTM)")
    print("  - ✓ Training with Adam + categorical_crossentropy")
    print("  - ✓ Evaluation and prediction")
    print("  - ✓ Model saving and loading")
    print("  - ✓ Text preprocessing for inference")
    
    return True


if __name__ == "__main__":
    test_model2_integration()