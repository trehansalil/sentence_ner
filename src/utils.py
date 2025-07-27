"""
Utility functions for the NER project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import json
import pickle
import os
from collections import Counter


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the NER dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Try different encodings to handle encoding issues
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, raise an error
    raise ValueError(f"Unable to read file {file_path} with any of the tried encodings: {encodings}")


def get_unique_tags(df: pd.DataFrame) -> List[str]:
    """
    Get unique NER tags from the dataset.
    
    Args:
        df (pd.DataFrame): Dataset containing 'Tag' column
        
    Returns:
        List[str]: List of unique tags
    """
    return sorted(df['Tag'].unique().tolist())


def get_tag_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get distribution of NER tags.
    
    Args:
        df (pd.DataFrame): Dataset containing 'Tag' column
        
    Returns:
        Dict[str, int]: Tag distribution
    """
    return dict(Counter(df['Tag']))


def plot_tag_distribution(tag_counts: Dict[str, int], title: str = "Tag Distribution"):
    """
    Plot distribution of NER tags.
    
    Args:
        tag_counts (Dict[str, int]): Tag distribution
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    tags = list(tag_counts.keys())
    counts = list(tag_counts.values())
    
    plt.bar(tags, counts)
    plt.title(title)
    plt.xlabel('NER Tags')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def create_tag_to_id_mapping(tags: List[str]) -> Dict[str, int]:
    """
    Create mapping from tags to numerical IDs.
    
    Args:
        tags (List[str]): List of unique tags
        
    Returns:
        Dict[str, int]: Tag to ID mapping
    """
    return {tag: idx for idx, tag in enumerate(tags)}


def create_id_to_tag_mapping(tag_to_id: Dict[str, int]) -> Dict[int, str]:
    """
    Create mapping from numerical IDs to tags.
    
    Args:
        tag_to_id (Dict[str, int]): Tag to ID mapping
        
    Returns:
        Dict[int, str]: ID to tag mapping
    """
    return {idx: tag for tag, idx in tag_to_id.items()}


def save_model(model: Any, filepath: str) -> None:
    """
    Save model to file.
    
    Args:
        model: Model object to save
        filepath (str): Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> Any:
    """
    Load model from file.
    
    Args:
        filepath (str): Path to the model file
        
    Returns:
        Any: Loaded model object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results (Dict[str, Any]): Results dictionary
        filepath (str): Path to save the results
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath (str): Path to the results file
        
    Returns:
        Dict[str, Any]: Loaded results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def print_dataset_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("Dataset Information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Number of unique sentences: {df['Sentence #'].nunique()}")
    print(f"Number of unique words: {df['Word'].nunique()}")
    print(f"Number of unique tags: {df['Tag'].nunique()}")
    print(f"Unique tags: {sorted(df['Tag'].unique())}")


def get_sentence_length_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get statistics about sentence lengths.
    
    Args:
        df (pd.DataFrame): Dataset containing sentences
        
    Returns:
        Dict[str, float]: Statistics about sentence lengths
    """
    sentence_lengths = df.groupby('Sentence #').size()
    
    return {
        'mean_length': sentence_lengths.mean(),
        'median_length': sentence_lengths.median(),
        'min_length': sentence_lengths.min(),
        'max_length': sentence_lengths.max(),
        'std_length': sentence_lengths.std()
    }


def plot_sentence_length_distribution(df: pd.DataFrame) -> None:
    """
    Plot distribution of sentence lengths.
    
    Args:
        df (pd.DataFrame): Dataset containing sentences
    """
    sentence_lengths = df.groupby('Sentence #').size()
    
    plt.figure(figsize=(12, 6))
    plt.hist(sentence_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Sentence Lengths')
    plt.xlabel('Sentence Length (number of words)')
    plt.ylabel('Frequency')
    plt.axvline(sentence_lengths.mean(), color='red', linestyle='--', label=f'Mean: {sentence_lengths.mean():.1f}')
    plt.axvline(sentence_lengths.median(), color='green', linestyle='--', label=f'Median: {sentence_lengths.median():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def extract_entities(words: List[str], tags: List[str]) -> List[Tuple[str, str]]:
    """
    Extract named entities from IOB2 tags.
    
    Args:
        words (List[str]): List of words
        tags (List[str]): List of corresponding IOB2 tags
        
    Returns:
        List[Tuple[str, str]]: List of (entity_text, entity_type) tuples
    """
    entities = []
    current_entity = []
    current_type = None
    
    for word, tag in zip(words, tags):
        if tag.startswith('B-'):
            # Save previous entity if exists
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
            # Start new entity
            current_entity = [word]
            current_type = tag[2:]
        elif tag.startswith('I-') and current_type == tag[2:]:
            # Continue current entity
            current_entity.append(word)
        else:
            # End current entity if exists
            if current_entity:
                entities.append((' '.join(current_entity), current_type))
            current_entity = []
            current_type = None
    
    # Handle last entity
    if current_entity:
        entities.append((' '.join(current_entity), current_type))
    
    return entities


def calculate_entity_level_metrics(true_entities: List[Tuple[str, str]], 
                                 pred_entities: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Calculate entity-level precision, recall, and F1-score.
    
    Args:
        true_entities (List[Tuple[str, str]]): True entities
        pred_entities (List[Tuple[str, str]]): Predicted entities
        
    Returns:
        Dict[str, float]: Entity-level metrics
    """
    true_set = set(true_entities)
    pred_set = set(pred_entities)
    
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }