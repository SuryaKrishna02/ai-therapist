import re
import json
import torch
import transformers
import seaborn as sns
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class TherapyResponseEvaluator:
    """
    A class for evaluating therapy response predictions against ground truth.
    Handles both classification metrics and response quality evaluation.
    """
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        """Initialize the evaluator with LLaMA model."""
        self.emotions = [
            "anger", "sadness", "disgust", "depression",
            "neutral", "joy", "fear"
        ]
        
        self.strategies = [
            "Open questions", "Approval", "Self-disclosure",
            "Restatement", "Interpretation", "Advisement",
            "Communication Skills", "Structuring the therapy",
            "Guiding the pace", "Others"
        ]
        
        # Create lowercase versions for matching
        self.emotions_lower = [e.lower() for e in self.emotions]
        self.strategies_lower = [s.lower() for s in self.strategies]
        
        # Initialize LLaMA model
        self.llama_pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for comparison.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation except hyphens in multi-word terms
        text = re.sub(r'[^\w\s-]', '', text)
        return text.strip()

    def normalize_prediction(self, pred: str, reference_list: List[str], 
                           reference_list_lower: List[str]) -> str:
        """
        Normalize prediction to match reference list.
        
        Args:
            pred (str): Prediction to normalize
            reference_list (List[str]): Original reference list
            reference_list_lower (List[str]): Lowercase reference list
            
        Returns:
            str: Normalized prediction
        """
        pred_processed = self.preprocess_text(pred)
        try:
            idx = reference_list_lower.index(pred_processed)
            return reference_list[idx]  # Return original case version
        except ValueError:
            # If no exact match, return the input (will be counted as error)
            return pred

    def calculate_classification_metrics(self, y_true: List[str], y_pred: List[str], 
                                      labels: List[str]) -> Dict[str, float]:
        """Calculate classification metrics with preprocessing."""
        # Normalize predictions and ground truth
        if labels == self.emotions:
            y_true_norm = [self.normalize_prediction(y, self.emotions, self.emotions_lower) for y in y_true]
            y_pred_norm = [self.normalize_prediction(y, self.emotions, self.emotions_lower) for y in y_pred]
        else:
            y_true_norm = [self.normalize_prediction(y, self.strategies, self.strategies_lower) for y in y_true]
            y_pred_norm = [self.normalize_prediction(y, self.strategies, self.strategies_lower) for y in y_pred]

        accuracy = accuracy_score(y_true_norm, y_pred_norm)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_norm, y_pred_norm, labels=labels, average='macro', zero_division=0
        )
        
        # Calculate per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y_true_norm, y_pred_norm, labels=labels, average=None, zero_division=0
        )
        
        per_class_metrics = {}
        for i, label in enumerate(labels):
            per_class_metrics[label] = {
                'precision': per_class_precision[i],
                'recall': per_class_recall[i],
                'f1': per_class_f1[i]
            }
        
        return {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'per_class': per_class_metrics
        }

    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                            labels: List[str], title: str) -> None:
        """Plot confusion matrix with preprocessed labels."""
        if labels == self.emotions:
            y_true_norm = [self.normalize_prediction(y, self.emotions, self.emotions_lower) for y in y_true]
            y_pred_norm = [self.normalize_prediction(y, self.emotions, self.emotions_lower) for y in y_pred]
        else:
            y_true_norm = [self.normalize_prediction(y, self.strategies, self.strategies_lower) for y in y_true]
            y_pred_norm = [self.normalize_prediction(y, self.strategies, self.strategies_lower) for y in y_pred]

        cm = confusion_matrix(y_true_norm, y_pred_norm, labels=labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='YlOrRd')
        plt.title(title, pad=20)
        plt.ylabel('True Label', labelpad=10)
        plt.xlabel('Predicted Label', labelpad=10)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    async def evaluate_response_quality(self, true_response: str, pred_response: str) -> float:
        """
        Use LLaMA to evaluate response quality with a single rating.
        
        Args:
            true_response (str): Ground truth therapist response
            pred_response (str): Predicted therapist response
            
        Returns:
            float: Overall quality rating (0-1)
        """
        system_instruction = """As an expert therapy evaluator, rate the following predicted therapist response compared to the ground truth response.
        Rate the overall quality on a scale of 1-10, considering:
        - Relevance to the therapeutic context
        - Coherence and clarity
        - Therapeutic effectiveness
        - Demonstration of empathy
        
        """
        prompt = f"""
        Ground Truth: "{true_response}"
        Predicted: "{pred_response}"
        
        Provide a brief explanation, then return only a single number between 1-10.
        """

        try:

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ]
            outputs = self.llama_pipeline(
                messages,
                max_new_tokens=100,
                temperature=0.1
            )

            response = outputs[0]['generated_text'][-1]['content'].strip()
            
            # Extract the numerical rating (last number in the response)
            numbers = re.findall(r'\d+', response)
            if numbers:
                rating = min(10, max(1, int(numbers[-1])))  # Ensure it's between 1-10
                return rating / 10.0
            return 0.0
            
        except Exception as e:
            print(f"Error in response evaluation: {str(e)}")
            return 0.0

    def print_metrics_table(self, metrics: Dict) -> None:
        """
        Print metrics in a nicely formatted table.
        
        Args:
            metrics (Dict): Dictionary containing evaluation metrics
        """
        # Overall metrics table
        overall_data = []
        for category in ['client_emotion', 'therapist_emotion', 'strategy']:
            row = [
                category.replace('_', ' ').title(),
                f"{metrics[category]['overall']['accuracy']:.3f}",
                f"{metrics[category]['overall']['precision']:.3f}",
                f"{metrics[category]['overall']['recall']:.3f}",
                f"{metrics[category]['overall']['f1']:.3f}"
            ]
            overall_data.append(row)

        print("\n=== Overall Metrics ===")
        print(tabulate(
            overall_data,
            headers=['Category', 'Accuracy', 'Precision', 'Recall', 'F1'],
            tablefmt='grid'
        ))

        # Per-class metrics tables
        for category in ['client_emotion', 'therapist_emotion', 'strategy']:
            print(f"\n=== {category.replace('_', ' ').title()} Per-Class Metrics ===")
            per_class_data = []
            for label, scores in metrics[category]['per_class'].items():
                row = [
                    label,
                    f"{scores['precision']:.3f}",
                    f"{scores['recall']:.3f}",
                    f"{scores['f1']:.3f}"
                ]
                per_class_data.append(row)

            print(tabulate(
                per_class_data,
                headers=['Class', 'Precision', 'Recall', 'F1'],
                tablefmt='grid'
            ))

    async def evaluate_all_metrics(self, ground_truth: List[Dict], predictions: List[Dict]) -> Tuple[Dict, List[float]]:
        """
        Calculate and display all metrics.
        
        Args:
            ground_truth (List[Dict]): List of ground truth entries
            predictions (List[Dict]): List of predicted entries
        """
        # Extract components
        client_emotions_true = [entry['client_emotion'] for entry in ground_truth]
        client_emotions_pred = [entry['client_emotion'] for entry in predictions]
        
        therapist_emotions_true = [entry['therapist_emotion'] for entry in ground_truth]
        therapist_emotions_pred = [entry['therapist_emotion'] for entry in predictions]
        
        strategies_true = [entry['strategy'] for entry in ground_truth]
        strategies_pred = [entry['strategy'] for entry in predictions]
        
        # Calculate classification metrics
        metrics = {
            'client_emotion': self.calculate_classification_metrics(
                client_emotions_true, client_emotions_pred, self.emotions
            ),
            'therapist_emotion': self.calculate_classification_metrics(
                therapist_emotions_true, therapist_emotions_pred, self.emotions
            ),
            'strategy': self.calculate_classification_metrics(
                strategies_true, strategies_pred, self.strategies
            )
        }
        
        # Print metrics in tables
        self.print_metrics_table(metrics)
        
        # Plot confusion matrices
        self.plot_confusion_matrix(
            client_emotions_true, client_emotions_pred, 
            self.emotions, "Client Emotion Confusion Matrix"
        )
        self.plot_confusion_matrix(
            therapist_emotions_true, therapist_emotions_pred, 
            self.emotions, "Therapist Emotion Confusion Matrix"
        )
        self.plot_confusion_matrix(
            strategies_true, strategies_pred, 
            self.strategies, "Therapeutic Strategy Confusion Matrix"
        )
        
        # Calculate response quality scores
        quality_scores = []
        for gt, pred in zip(ground_truth, predictions):
            score = await self.evaluate_response_quality(gt['therapist'], pred['therapist'])
            quality_scores.append(score)
        
        # Print response quality metrics
        print("\n=== Response Quality Metrics ===")
        print(tabulate([
            ['Average Quality Score', f"{sum(quality_scores) / len(quality_scores):.3f}"],
            ['Min Quality Score', f"{min(quality_scores):.3f}"],
            ['Max Quality Score', f"{max(quality_scores):.3f}"]
        ], tablefmt='grid'))

        return metrics, quality_scores
    
    def save_metrics_to_json(self, metrics: Dict, response_quality_scores: List[float], output_file: str) -> None:
        """
        Save evaluation metrics and response quality scores to a JSON file.
        
        Args:
            metrics (Dict): Dictionary containing all evaluation metrics
            response_quality_scores (List[float]): List of response quality scores
            output_file (str): Path to output JSON file
        """
        # Prepare results dictionary
        results = {
            'classification_metrics': metrics,
            'response_quality': {
                'scores': response_quality_scores,
                'average': sum(response_quality_scores) / len(response_quality_scores),
                'min': min(response_quality_scores),
                'max': max(response_quality_scores)
            },
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"\nMetrics saved to: {output_file}")
        except Exception as e:
            print(f"Error saving metrics to file: {str(e)}")