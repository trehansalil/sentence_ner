"""
Evaluation module for NER models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_fscore_support,
    accuracy_score
)
from typing import List, Dict, Any, Tuple
import pandas as pd
from .utils import extract_entities, calculate_entity_level_metrics


class NERModelEvaluator:
    """Evaluator for NER models."""
    
    def __init__(self, id_to_tag: Dict[int, str]):
        """
        Initialize the evaluator.
        
        Args:
            id_to_tag (Dict[int, str]): Mapping from tag IDs to tag names
        """
        self.id_to_tag = id_to_tag
        self.tag_to_id = {tag: idx for idx, tag in id_to_tag.items()}
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      X_test: np.ndarray = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of NER model.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            X_test (np.ndarray, optional): Test input sequences
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        results = {}
        
        # Flatten sequences for token-level evaluation
        y_true_flat = self._flatten_sequences(y_true)
        y_pred_flat = self._flatten_sequences(y_pred)
        
        # Token-level metrics
        results['token_level'] = self._calculate_token_level_metrics(y_true_flat, y_pred_flat)
        
        # Sequence-level metrics
        results['sequence_level'] = self._calculate_sequence_level_metrics(y_true, y_pred)
        
        # Entity-level metrics
        results['entity_level'] = self._calculate_entity_level_metrics(y_true, y_pred)
        
        # Per-tag metrics
        results['per_tag'] = self._calculate_per_tag_metrics(y_true_flat, y_pred_flat)
        
        # Confusion matrix
        results['confusion_matrix'] = self._calculate_confusion_matrix(y_true_flat, y_pred_flat)
        
        return results
    
    def _flatten_sequences(self, sequences: np.ndarray) -> List[int]:
        """
        Flatten padded sequences, removing padding tokens.
        
        Args:
            sequences (np.ndarray): Padded sequences
            
        Returns:
            List[int]: Flattened sequences without padding
        """
        flattened = []
        for seq in sequences:
            for token in seq:
                # Assuming 'O' tag has ID 0 and is used for padding
                if token != self.tag_to_id.get('O', 0) or len(flattened) == 0:
                    flattened.append(token)
                elif token == self.tag_to_id.get('O', 0):
                    # Only add 'O' if it's not padding (i.e., if the previous token wasn't 'O')
                    if flattened and flattened[-1] != self.tag_to_id.get('O', 0):
                        flattened.append(token)
        return flattened
    
    def _calculate_token_level_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Calculate token-level metrics.
        
        Args:
            y_true (List[int]): True labels (flattened)
            y_pred (List[int]): Predicted labels (flattened)
            
        Returns:
            Dict[str, float]: Token-level metrics
        """
        # Ensure equal length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def _calculate_sequence_level_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate sequence-level metrics.
        
        Args:
            y_true (np.ndarray): True label sequences
            y_pred (np.ndarray): Predicted label sequences
            
        Returns:
            Dict[str, float]: Sequence-level metrics
        """
        exact_matches = 0
        total_sequences = len(y_true)
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            if np.array_equal(true_seq, pred_seq):
                exact_matches += 1
        
        sequence_accuracy = exact_matches / total_sequences
        
        return {
            'sequence_accuracy': sequence_accuracy,
            'exact_matches': exact_matches,
            'total_sequences': total_sequences
        }
    
    def _calculate_entity_level_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate entity-level metrics.
        
        Args:
            y_true (np.ndarray): True label sequences
            y_pred (np.ndarray): Predicted label sequences
            
        Returns:
            Dict[str, Any]: Entity-level metrics
        """
        all_true_entities = []
        all_pred_entities = []
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            # Convert to tag names
            true_tags = [self.id_to_tag[tag_id] for tag_id in true_seq]
            pred_tags = [self.id_to_tag[tag_id] for tag_id in pred_seq]
            
            # Create dummy words for entity extraction
            words = [f"word_{i}" for i in range(len(true_tags))]
            
            # Extract entities
            true_entities = extract_entities(words, true_tags)
            pred_entities = extract_entities(words, pred_tags)
            
            all_true_entities.extend(true_entities)
            all_pred_entities.extend(pred_entities)
        
        # Calculate entity-level metrics
        return calculate_entity_level_metrics(all_true_entities, all_pred_entities)
    
    def _calculate_per_tag_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each tag.
        
        Args:
            y_true (List[int]): True labels (flattened)
            y_pred (List[int]): Predicted labels (flattened)
            
        Returns:
            Dict[str, Dict[str, float]]: Per-tag metrics
        """
        # Ensure equal length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Get unique tags
        unique_tags = list(self.id_to_tag.keys())
        
        # Calculate per-tag metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_tags, average=None, zero_division=0
        )
        
        per_tag_metrics = {}
        for i, tag_id in enumerate(unique_tags):
            tag_name = self.id_to_tag[tag_id]
            per_tag_metrics[tag_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i])
            }
        
        return per_tag_metrics
    
    def _calculate_confusion_matrix(self, y_true: List[int], y_pred: List[int]) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true (List[int]): True labels (flattened)
            y_pred (List[int]): Predicted labels (flattened)
            
        Returns:
            np.ndarray: Confusion matrix
        """
        # Ensure equal length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        return confusion_matrix(y_true, y_pred)
    
    def plot_confusion_matrix(self, cm: np.ndarray, title: str = "Confusion Matrix") -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            title (str): Plot title
        """
        plt.figure(figsize=(12, 10))
        
        # Get tag names for labels
        tag_names = [self.id_to_tag[i] for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=tag_names, yticklabels=tag_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_per_tag_metrics(self, per_tag_metrics: Dict[str, Dict[str, float]], 
                           metric: str = 'f1_score') -> None:
        """
        Plot per-tag metrics.
        
        Args:
            per_tag_metrics (Dict[str, Dict[str, float]]): Per-tag metrics
            metric (str): Metric to plot ('precision', 'recall', 'f1_score')
        """
        tags = list(per_tag_metrics.keys())
        values = [per_tag_metrics[tag][metric] for tag in tags]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(tags, values)
        plt.title(f'Per-Tag {metric.replace("_", " ").title()}')
        plt.xlabel('NER Tags')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def generate_classification_report(self, y_true: List[int], y_pred: List[int]) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true (List[int]): True labels (flattened)
            y_pred (List[int]): Predicted labels (flattened)
            
        Returns:
            str: Classification report
        """
        # Ensure equal length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        target_names = [self.id_to_tag[i] for i in sorted(self.id_to_tag.keys())]
        
        return classification_report(
            y_true, y_pred, 
            target_names=target_names, 
            zero_division=0
        )
    
    def compare_models(self, results_list: List[Dict[str, Any]], 
                      model_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple model results.
        
        Args:
            results_list (List[Dict[str, Any]]): List of evaluation results
            model_names (List[str]): Names of the models
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for results, name in zip(results_list, model_names):
            row = {
                'Model': name,
                'Token Accuracy': results['token_level']['accuracy'],
                'Token F1': results['token_level']['f1_score'],
                'Token Precision': results['token_level']['precision'],
                'Token Recall': results['token_level']['recall'],
                'Sequence Accuracy': results['sequence_level']['sequence_accuracy'],
                'Entity F1': results['entity_level']['f1_score'],
                'Entity Precision': results['entity_level']['precision'],
                'Entity Recall': results['entity_level']['recall']
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """
        Plot model comparison.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
        """
        metrics = ['Token F1', 'Token Accuracy', 'Sequence Accuracy', 'Entity F1']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric])
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(comparison_df['Model'].iloc[0]) > 10:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        id_to_tag: Dict[int, str]) -> Dict[str, Any]:
    """
    Convenience function to evaluate model predictions.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        id_to_tag (Dict[int, str]): Tag ID to name mapping
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    evaluator = NERModelEvaluator(id_to_tag)
    return evaluator.evaluate_model(y_true, y_pred)