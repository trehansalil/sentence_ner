"""
Baseline NER model implementation using feedforward neural network.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, Dense, Dropout, Flatten, Input,
    GlobalMaxPooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from .utils import save_model, save_results
import os


class BaselineNERModel:
    """Baseline NER model using feedforward neural network."""
    
    def __init__(self, vocab_size: int, num_tags: int, max_sequence_length: int,
                 embedding_dim: int = 100, hidden_dim: int = 128, dropout_rate: float = 0.3):
        """
        Initialize the baseline model.
        
        Args:
            vocab_size (int): Size of vocabulary
            num_tags (int): Number of NER tags
            max_sequence_length (int): Maximum sequence length
            embedding_dim (int): Embedding dimension
            hidden_dim (int): Hidden layer dimension
            dropout_rate (float): Dropout rate
        """
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the baseline model architecture.
        
        Returns:
            Model: Compiled Keras model
        """
        # Input layer
        input_layer = Input(shape=(self.max_sequence_length,), name='input_tokens')
        
        # Embedding layer
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            mask_zero=True,
            name='word_embedding'
        )(input_layer)
        
        # Global max pooling to get fixed-size representation
        pooled = GlobalMaxPooling1D()(embedding)
        
        # Hidden layers
        hidden1 = Dense(self.hidden_dim, activation='relu', name='hidden1')(pooled)
        dropout1 = Dropout(self.dropout_rate)(hidden1)
        
        hidden2 = Dense(self.hidden_dim // 2, activation='relu', name='hidden2')(dropout1)
        dropout2 = Dropout(self.dropout_rate)(hidden2)
        
        # Output layer - we need to predict for each token in the sequence
        # So we'll repeat the representation and use TimeDistributed
        from tensorflow.keras.layers import RepeatVector, TimeDistributed
        
        # Repeat the pooled representation for each time step
        repeated = RepeatVector(self.max_sequence_length)(dropout2)
        
        # Dense layer for each time step
        output = TimeDistributed(
            Dense(self.num_tags, activation='softmax'), 
            name='tag_predictions'
        )(repeated)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_simple_model(self) -> Model:
        """
        Build a simpler baseline model that works token by token.
        
        Returns:
            Model: Compiled Keras model
        """
        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                mask_zero=True
            ),
            
            # Flatten to work with each token independently
            Flatten(),
            
            # Hidden layers
            Dense(self.hidden_dim, activation='relu'),
            Dropout(self.dropout_rate),
            
            Dense(self.hidden_dim // 2, activation='relu'),
            Dropout(self.dropout_rate),
            
            # Reshape back to sequence format
            Dense(self.max_sequence_length * self.num_tags, activation='linear'),
            tf.keras.layers.Reshape((self.max_sequence_length, self.num_tags)),
            
            # Softmax activation
            tf.keras.layers.Activation('softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              patience: int = 10, model_save_path: str = None) -> Dict[str, Any]:
        """
        Train the baseline model.
        
        Args:
            X_train (np.ndarray): Training input sequences
            y_train (np.ndarray): Training target sequences
            X_val (np.ndarray): Validation input sequences
            y_val (np.ndarray): Validation target sequences
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            patience (int): Early stopping patience
            model_save_path (str): Path to save the best model
            
        Returns:
            Dict[str, Any]: Training history and metrics
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    model_save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Return training summary
        return {
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_train_accuracy': float(self.history.history['accuracy'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'best_val_loss': float(min(self.history.history['val_loss'])),
            'best_val_accuracy': float(max(self.history.history['val_accuracy'])),
            'epochs_trained': len(self.history.history['loss'])
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input sequences.
        
        Args:
            X (np.ndarray): Input sequences
            
        Returns:
            np.ndarray: Predicted tag sequences
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet.")
        
        # Get probabilities
        probabilities = self.model.predict(X)
        
        # Convert to predicted classes
        predictions = np.argmax(probabilities, axis=-1)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test input sequences
            y_test (np.ndarray): Test target sequences
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet.")
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy)
        }
    
    def plot_training_history(self, save_path: str = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "Model not built yet."
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)


def create_baseline_model(vocab_size: int, num_tags: int, max_sequence_length: int,
                         **kwargs) -> BaselineNERModel:
    """
    Factory function to create a baseline NER model.
    
    Args:
        vocab_size (int): Size of vocabulary
        num_tags (int): Number of NER tags
        max_sequence_length (int): Maximum sequence length
        **kwargs: Additional model parameters
        
    Returns:
        BaselineNERModel: Initialized baseline model
    """
    return BaselineNERModel(
        vocab_size=vocab_size,
        num_tags=num_tags,
        max_sequence_length=max_sequence_length,
        **kwargs
    )


def train_baseline_model(model: BaselineNERModel, 
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        model_save_path: str = "models/baseline_model.h5",
                        results_save_path: str = "results/baseline_results.json",
                        **kwargs) -> Dict[str, Any]:
    """
    Train a baseline model and save results.
    
    Args:
        model (BaselineNERModel): Model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_save_path (str): Path to save the model
        results_save_path (str): Path to save results
        **kwargs: Additional training parameters
        
    Returns:
        Dict[str, Any]: Training results
    """
    # Train the model
    training_results = model.train(
        X_train, y_train, X_val, y_val,
        model_save_path=model_save_path,
        **kwargs
    )
    
    # Save results
    save_results(training_results, results_save_path)
    
    return training_results