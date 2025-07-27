"""
Data preprocessing module for NER project.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
from .utils import (
    get_unique_tags, 
    create_tag_to_id_mapping, 
    create_id_to_tag_mapping,
    save_results
)


class NERDataProcessor:
    """Data processor for NER dataset."""
    
    def __init__(self, max_sequence_length: int = 128):
        """
        Initialize the data processor.
        
        Args:
            max_sequence_length (int): Maximum sequence length for padding
        """
        self.max_sequence_length = max_sequence_length
        self.tag_to_id = {}
        self.id_to_tag = {}
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.num_tags = 0
        
    def load_data(self, file_path):
        """
        Loads the dataset from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            # Try reading with a different encoding
            self.df = pd.read_csv(file_path, encoding='latin-1')
        except UnicodeDecodeError:
            # Fallback to default encoding if latin-1 fails
            self.df = pd.read_csv(file_path, encoding='iso-8859-1')
        return self.df
    
    def create_sentences(self, df: pd.DataFrame) -> List[Tuple[List[str], List[str]]]:
        """
        Group words and tags by sentence.
        
        Args:
            df (pd.DataFrame): Dataset with Sentence #, Word, and Tag columns
            
        Returns:
            List[Tuple[List[str], List[str]]]: List of (words, tags) tuples
        """
        sentences = []
        
        for sentence_id in df['Sentence #'].unique():
            sentence_data = df[df['Sentence #'] == sentence_id]
            words = sentence_data['Word'].tolist()
            tags = sentence_data['Tag'].tolist()
            sentences.append((words, tags))
            
        return sentences
    
    def build_vocabularies(self, sentences: List[Tuple[List[str], List[str]]]) -> None:
        """
        Build word and tag vocabularies.
        
        Args:
            sentences: List of (words, tags) tuples
        """
        # Build word vocabulary
        all_words = []
        all_tags = []
        
        for words, tags in sentences:
            all_words.extend(words)
            all_tags.extend(tags)
        
        # Create word mappings (including UNK token)
        unique_words = ['<PAD>', '<UNK>'] + sorted(list(set(all_words)))
        self.word_to_id = {word: idx for idx, word in enumerate(unique_words)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.vocab_size = len(unique_words)
        
        # Create tag mappings
        unique_tags = sorted(list(set(all_tags)))
        self.tag_to_id = create_tag_to_id_mapping(unique_tags)
        self.id_to_tag = create_id_to_tag_mapping(self.tag_to_id)
        self.num_tags = len(unique_tags)
    
    def encode_sequences(self, sentences: List[Tuple[List[str], List[str]]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode words and tags to numerical sequences.
        
        Args:
            sentences: List of (words, tags) tuples
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Encoded word sequences and tag sequences
        """
        word_sequences = []
        tag_sequences = []
        
        for words, tags in sentences:
            # Encode words (use UNK for unknown words)
            word_seq = [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in words]
            tag_seq = [self.tag_to_id[tag] for tag in tags]
            
            word_sequences.append(word_seq)
            tag_sequences.append(tag_seq)
        
        # Pad sequences
        word_sequences = pad_sequences(
            word_sequences, 
            maxlen=self.max_sequence_length, 
            padding='post', 
            value=self.word_to_id['<PAD>']
        )
        tag_sequences = pad_sequences(
            tag_sequences, 
            maxlen=self.max_sequence_length, 
            padding='post', 
            value=self.tag_to_id['O']  # Pad with 'O' tag
        )
        
        return word_sequences, tag_sequences
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_size: float = 0.6, val_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (np.ndarray): Input sequences
            y (np.ndarray): Target sequences
            train_size (float): Proportion of data for training
            val_size (float): Proportion of data for validation
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple[np.ndarray, ...]: X_train, X_val, X_test, y_train, y_val, y_test
        """
        test_size = 1 - train_size - val_size
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def process_data(self, file_path: str) -> Dict[str, Any]:
        """
        Complete data processing pipeline.
        
        Args:
            file_path (str): Path to the dataset file
            
        Returns:
            Dict[str, Any]: Processed data splits and metadata
        """
        # Load data
        df = self.load_data(file_path)
        
        # Create sentences
        sentences = self.create_sentences(df)
        
        # Build vocabularies
        self.build_vocabularies(sentences)
        
        # Encode sequences
        X, y = self.encode_sequences(sentences)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Prepare metadata
        metadata = {
            'vocab_size': self.vocab_size,
            'num_tags': self.num_tags,
            'max_sequence_length': self.max_sequence_length,
            'tag_to_id': self.tag_to_id,
            'id_to_tag': self.id_to_tag,
            'word_to_id': dict(list(self.word_to_id.items())[:100]),  # Sample for storage
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'total_sentences': len(sentences)
        }
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'metadata': metadata,
            'sentences': sentences[:100]  # Sample sentences for inspection
        }
    
    def save_processor(self, filepath: str) -> None:
        """
        Save the data processor state.
        
        Args:
            filepath (str): Path to save the processor
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        processor_state = {
            'max_sequence_length': self.max_sequence_length,
            'tag_to_id': self.tag_to_id,
            'id_to_tag': self.id_to_tag,
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'vocab_size': self.vocab_size,
            'num_tags': self.num_tags
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(processor_state, f)
    
    def load_processor(self, filepath: str) -> None:
        """
        Load the data processor state.
        
        Args:
            filepath (str): Path to load the processor from
        """
        with open(filepath, 'rb') as f:
            processor_state = pickle.load(f)
        
        self.max_sequence_length = processor_state['max_sequence_length']
        self.tag_to_id = processor_state['tag_to_id']
        self.id_to_tag = processor_state['id_to_tag']
        self.word_to_id = processor_state['word_to_id']
        self.id_to_word = processor_state['id_to_word']
        self.vocab_size = processor_state['vocab_size']
        self.num_tags = processor_state['num_tags']
    
    def preprocess_text(self, text: str) -> np.ndarray:
        """
        Preprocess a single text string for prediction.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Preprocessed sequence
        """
        words = text.split()
        word_seq = [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in words]
        
        # Pad sequence
        word_seq = pad_sequences(
            [word_seq], 
            maxlen=self.max_sequence_length, 
            padding='post', 
            value=self.word_to_id['<PAD>']
        )
        
        return word_seq
    
    def decode_predictions(self, predictions: np.ndarray) -> List[str]:
        """
        Decode numerical predictions to tag labels.
        
        Args:
            predictions (np.ndarray): Numerical predictions
            
        Returns:
            List[str]: Decoded tag labels
        """
        return [self.id_to_tag[pred] for pred in predictions]


def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the dataset and return statistics.
    
    Args:
        df (pd.DataFrame): NER dataset
        
    Returns:
        Dict[str, Any]: Dataset analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['total_tokens'] = len(df)
    analysis['unique_sentences'] = df['Sentence #'].nunique()
    analysis['unique_words'] = df['Word'].nunique()
    analysis['unique_tags'] = df['Tag'].nunique()
    
    # Tag distribution
    tag_counts = df['Tag'].value_counts().to_dict()
    analysis['tag_distribution'] = tag_counts
    
    # Sentence length statistics
    sentence_lengths = df.groupby('Sentence #').size()
    analysis['sentence_length_stats'] = {
        'mean': float(sentence_lengths.mean()),
        'median': float(sentence_lengths.median()),
        'min': int(sentence_lengths.min()),
        'max': int(sentence_lengths.max()),
        'std': float(sentence_lengths.std())
    }
    
    # Entity type distribution
    entity_tags = [tag for tag in df['Tag'].unique() if tag != 'O']
    entity_type_counts = {}
    for tag in entity_tags:
        if tag.startswith('B-'):
            entity_type = tag[2:]
            b_count = (df['Tag'] == tag).sum()
            i_count = (df['Tag'] == f'I-{entity_type}').sum()
            entity_type_counts[entity_type] = b_count + i_count
    
    analysis['entity_type_distribution'] = entity_type_counts
    
    return analysis