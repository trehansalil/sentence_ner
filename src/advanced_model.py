"""
Advanced NER model implementation using BiLSTM with attention mechanism.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense, Dropout,
    TimeDistributed, Attention, MultiHeadAttention,
    LayerNormalization, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from .utils import save_model, save_results
import os


class AdvancedNERModel:
    """Advanced NER model using BiLSTM with attention mechanism."""
    
    def __init__(self, vocab_size: int, num_tags: int, max_sequence_length: int,
                 embedding_dim: int = 200, lstm_units: int = 128, 
                 num_attention_heads: int = 8, dropout_rate: float = 0.3,
                 use_crf: bool = False):
        """
        Initialize the advanced model.
        
        Args:
            vocab_size (int): Size of vocabulary
            num_tags (int): Number of NER tags
            max_sequence_length (int): Maximum sequence length
            embedding_dim (int): Embedding dimension
            lstm_units (int): LSTM units
            num_attention_heads (int): Number of attention heads
            dropout_rate (float): Dropout rate
            use_crf (bool): Whether to use CRF layer (requires tensorflow-addons)
        """
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.use_crf = use_crf
        self.model = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build the advanced model architecture.
        
        Returns:
            Model: Compiled Keras model
        """
        # Input layer
        input_layer = Input(shape=(self.max_sequence_length,), name='input_tokens')
        
        # Embedding layer with larger dimension
        embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length,
            mask_zero=True,
            name='word_embedding'
        )(input_layer)
        
        # Add dropout to embeddings
        embedding_dropout = Dropout(self.dropout_rate)(embedding)
        
        # First BiLSTM layer
        bilstm1 = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate),
            name='bilstm1'
        )(embedding_dropout)
        
        # Second BiLSTM layer
        bilstm2 = Bidirectional(
            LSTM(self.lstm_units // 2, return_sequences=True, dropout=self.dropout_rate),
            name='bilstm2'
        )(bilstm1)
        
        # Multi-head attention mechanism
        attention = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.lstm_units,
            name='multi_head_attention'
        )(bilstm2, bilstm2)
        
        # Add & Norm
        attention_add = Add()([bilstm2, attention])
        attention_norm = LayerNormalization()(attention_add)
        
        # Dropout after attention
        attention_dropout = Dropout(self.dropout_rate)(attention_norm)
        
        # Feed-forward network
        ffn = TimeDistributed(
            Dense(self.lstm_units * 2, activation='relu'),
            name='ffn1'
        )(attention_dropout)
        
        ffn_dropout = Dropout(self.dropout_rate)(ffn)
        
        ffn2 = TimeDistributed(
            Dense(self.lstm_units, activation='relu'),
            name='ffn2'
        )(ffn_dropout)
        
        # Add & Norm
        ffn_add = Add()([attention_dropout, ffn2])
        ffn_norm = LayerNormalization()(ffn_add)
        
        # Output layer
        if self.use_crf:
            # CRF layer (requires tensorflow-addons)
            try:
                import tensorflow_addons as tfa
                dense_output = TimeDistributed(
                    Dense(self.num_tags, activation='linear'),
                    name='dense_output'
                )(ffn_norm)
                
                crf = tfa.layers.CRF(self.num_tags, name='crf_layer')
                output = crf(dense_output)
                
                # Create model
                model = Model(inputs=input_layer, outputs=output)
                
                # Compile with CRF loss
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss=crf.loss,
                    metrics=[crf.accuracy]
                )
                
            except ImportError:
                print("tensorflow-addons not available, using regular dense layer instead of CRF")
                self.use_crf = False
        
        if not self.use_crf:
            # Regular dense layer with softmax
            output = TimeDistributed(
                Dense(self.num_tags, activation='softmax'),
                name='tag_predictions'
            )(ffn_norm)
            
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
    
    def build_simple_bilstm_model(self) -> Model:
        """
        Build a simpler BiLSTM model without attention.
        
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
        
        # Dropout on embeddings
        embedding_dropout = Dropout(self.dropout_rate)(embedding)
        
        # BiLSTM layers
        bilstm1 = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout_rate, 
                 recurrent_dropout=self.dropout_rate),
            name='bilstm1'
        )(embedding_dropout)
        
        bilstm2 = Bidirectional(
            LSTM(self.lstm_units // 2, return_sequences=True, dropout=self.dropout_rate,
                 recurrent_dropout=self.dropout_rate),
            name='bilstm2'
        )(bilstm1)
        
        # Dense layers
        dense1 = TimeDistributed(
            Dense(self.lstm_units, activation='relu'),
            name='dense1'
        )(bilstm2)
        
        dropout1 = Dropout(self.dropout_rate)(dense1)
        
        # Output layer
        output = TimeDistributed(
            Dense(self.num_tags, activation='softmax'),
            name='tag_predictions'
        )(dropout1)
        
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
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              patience: int = 15, model_save_path: str = None) -> Dict[str, Any]:
        """
        Train the advanced model.
        
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
            self.build_simple_bilstm_model()  # Use simpler model for stability
        
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
                patience=8,
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


class TransformerNERModel:
    """Transformer-based NER model using pre-trained embeddings."""
    
    def __init__(self, vocab_size: int, num_tags: int, max_sequence_length: int,
                 embedding_dim: int = 256, num_heads: int = 8, 
                 num_transformer_blocks: int = 4, ff_dim: int = 512,
                 dropout_rate: float = 0.1):
        """
        Initialize the transformer model.
        
        Args:
            vocab_size (int): Size of vocabulary
            num_tags (int): Number of NER tags
            max_sequence_length (int): Maximum sequence length
            embedding_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            num_transformer_blocks (int): Number of transformer blocks
            ff_dim (int): Feed-forward dimension
            dropout_rate (float): Dropout rate
        """
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
    
    def transformer_block(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        """
        Create a transformer block.
        
        Args:
            inputs: Input tensor
            head_size: Size of attention heads
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            
        Returns:
            Output tensor
        """
        # Multi-head attention
        attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size
        )(inputs, inputs)
        attention = Dropout(dropout)(attention)
        attention = Add()([inputs, attention])
        attention = LayerNormalization()(attention)
        
        # Feed-forward network
        ffn = Dense(ff_dim, activation='relu')(attention)
        ffn = Dropout(dropout)(ffn)
        ffn = Dense(inputs.shape[-1])(ffn)
        ffn = Dropout(dropout)(ffn)
        ffn = Add()([attention, ffn])
        ffn = LayerNormalization()(ffn)
        
        return ffn


def create_advanced_model(vocab_size: int, num_tags: int, max_sequence_length: int,
                         **kwargs) -> AdvancedNERModel:
    """
    Factory function to create an advanced NER model.
    
    Args:
        vocab_size (int): Size of vocabulary
        num_tags (int): Number of NER tags
        max_sequence_length (int): Maximum sequence length
        **kwargs: Additional model parameters
        
    Returns:
        AdvancedNERModel: Initialized advanced model
    """
    return AdvancedNERModel(
        vocab_size=vocab_size,
        num_tags=num_tags,
        max_sequence_length=max_sequence_length,
        **kwargs
    )


def train_advanced_model(model: AdvancedNERModel, 
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        model_save_path: str = "models/advanced_model.h5",
                        results_save_path: str = "results/advanced_results.json",
                        **kwargs) -> Dict[str, Any]:
    """
    Train an advanced model and save results.
    
    Args:
        model (AdvancedNERModel): Model to train
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