#!/usr/bin/env python3
"""
Main script demonstrating Model 2 NER implementation.
"""

import os
from src.data_preprocessing import NERDataProcessor
from src.advanced_model import Model2NER, create_model2_ner


def main():
    """Main function demonstrating Model 2 NER pipeline."""
    print("Sentence NER - Model 2 Implementation")
    print("=" * 50)
    
    # Data preprocessing
    print("1. Data Preprocessing")
    processor = NERDataProcessor(max_sequence_length=75)  # Match notebook setting
    
    data_path = "data/ner_dataset.csv"
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        processed_data = processor.process_data(data_path, categorical_tags=True)
        
        print(f"✓ Data processed successfully:")
        print(f"  - Vocabulary size: {processed_data['metadata']['vocab_size']:,}")
        print(f"  - Number of tags: {processed_data['metadata']['num_tags']}")
        print(f"  - Training samples: {processed_data['metadata']['train_size']:,}")
        print(f"  - Validation samples: {processed_data['metadata']['val_size']:,}")
        print(f"  - Test samples: {processed_data['metadata']['test_size']:,}")
    else:
        print(f"Data file not found at {data_path}")
        return
    
    # Model creation and training
    print("\n2. Model 2 Creation and Training")
    model = create_model2_ner(
        vocab_size=processed_data['metadata']['vocab_size'],
        num_tags=processed_data['metadata']['num_tags'],
        max_sequence_length=processed_data['metadata']['max_sequence_length']
    )
    
    print("✓ Model 2 created with parameters:")
    print(f"  - Embedding dimension: 50")
    print(f"  - LSTM units: 100")
    print(f"  - Recurrent dropout: 0.1")
    
    # Build model to see architecture
    keras_model = model.build_model()
    keras_model.build(input_shape=(None, processed_data['metadata']['max_sequence_length']))
    print(f"  - Total parameters: {keras_model.count_params():,}")
    
    # Training with notebook parameters
    print("\n3. Training Model 2 (Notebook Parameters)")
    print("Training with parameters from notebook:")
    print("  - Epochs: 10")
    print("  - Batch size: 64")
    print("  - Optimizer: Adam")
    print("  - Loss: categorical_crossentropy")
    
    history = model.train(
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data['X_val'],
        processed_data['y_val'],
        epochs=10,
        batch_size=64,
        verbose=1
    )
    
    print("✓ Training completed!")
    print(f"  - Final train accuracy: {history['final_train_accuracy']:.4f}")
    print(f"  - Final validation accuracy: {history['final_val_accuracy']:.4f}")
    print(f"  - Best validation accuracy: {history['best_val_accuracy']:.4f}")
    
    # Evaluation
    print("\n4. Model Evaluation")
    test_results = model.evaluate(
        processed_data['X_test'],
        processed_data['y_test']
    )
    
    print(f"✓ Test Results:")
    print(f"  - Test accuracy: {test_results['test_accuracy']:.4f}")
    print(f"  - Test loss: {test_results['test_loss']:.4f}")
    
    # Save model and results
    print("\n5. Saving Model and Results")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    model.save_model("models/model2_ner.h5")
    
    # Save comprehensive results
    final_results = {
        **history,
        **test_results,
        'model_type': 'Model2NER',
        'model_parameters': {
            'vocab_size': processed_data['metadata']['vocab_size'],
            'num_tags': processed_data['metadata']['num_tags'],
            'max_sequence_length': processed_data['metadata']['max_sequence_length'],
            'embedding_dim': 50,
            'lstm_units': 100,
            'recurrent_dropout': 0.1
        }
    }
    
    import json
    with open("results/model2_final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("✓ Model and results saved:")
    print("  - Model: models/model2_ner.h5")
    print("  - Results: results/model2_final_results.json")
    
    # Plot training history
    try:
        model.plot_training_history(save_path="results/model2_training_history.png")
        print("  - Training plots: results/model2_training_history.png")
    except Exception as e:
        print(f"  - Note: Could not save plots: {e}")
    
    print("\n" + "=" * 50)
    print("Model 2 NER implementation completed successfully!")
    print("This implementation matches the Model 2 from NER_Prediction_Final.ipynb")


if __name__ == "__main__":
    main()
