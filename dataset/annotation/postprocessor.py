"""
This file contains the code for the postprocessing of the annotated data.
"""

import pandas as pd
from typing import List, Dict

class AnnotationPostprocessor:
    def __init__(self):
        self.emotion_categories = ['anger', 'sadness', 'disgust', 'depression', 'neutral', 'joy', 'fear']
        self.strategy_categories = [
            'Open questions', 'Approval', 'Self-disclosure', 'Restatement',
            'Interpretation', 'Advisement', 'Communication Skills',
            'Structuring the therapy', 'Guiding the pace'
        ]

    def process_annotations(self, annotations: List[Dict]) -> Dict:
        """Process and analyze annotation results."""
        df = pd.DataFrame(annotations)
        
        # Emotion analysis
        emotion_summary = self._analyze_emotions(df)
        
        # Strategy analysis
        strategy_summary = self._analyze_strategies(df)
        
        # Temporal analysis
        temporal_patterns = self._analyze_temporal_patterns(df)
        
        return {
            'emotion_summary': emotion_summary,
            'strategy_summary': strategy_summary,
            'temporal_patterns': temporal_patterns
        }

    def _analyze_emotions(self, df: pd.DataFrame) -> Dict:
        """Analyze emotion patterns and transitions."""
        emotion_counts = df['client_emotion'].value_counts().to_dict()
        emotion_transitions = self._calculate_transitions(df['client_emotion'])
        
        return {
            'distribution': emotion_counts,
            'transitions': emotion_transitions
        }

    def _analyze_strategies(self, df: pd.DataFrame) -> Dict:
        """Analyze therapeutic strategy patterns."""
        strategy_counts = df['therapist_strategy'].value_counts().to_dict()
        strategy_effectiveness = self._evaluate_strategy_effectiveness(df)
        
        return {
            'distribution': strategy_counts,
            'effectiveness': strategy_effectiveness
        }

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns over time."""
        # Add timestamp-based analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Emotion progression
        emotion_progression = df.groupby(pd.Grouper(key='timestamp', freq='5min'))['client_emotion'].agg(lambda x: x.mode()[0])
        
        # Strategy adaptation
        strategy_adaptation = df.groupby(pd.Grouper(key='timestamp', freq='5min'))['therapist_strategy'].agg(lambda x: x.mode()[0])
        
        return {
            'emotion_progression': emotion_progression.to_dict(),
            'strategy_adaptation': strategy_adaptation.to_dict()
        }

    def _calculate_transitions(self, series: pd.Series) -> Dict:
        """Calculate transition probabilities between states."""
        transitions = {}
        previous_state = None
        
        for state in series:
            if previous_state is not None:
                transition_key = f"{previous_state}->{state}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1
            previous_state = state
            
        return transitions

    def _evaluate_strategy_effectiveness(self, df: pd.DataFrame) -> Dict:
        """Evaluate effectiveness of therapeutic strategies."""
        effectiveness = {}
        
        for strategy in self.strategy_categories:
            # Consider a strategy effective if it leads to positive emotion changes
            strategy_segments = df[df['therapist_strategy'] == strategy]
            
            if len(strategy_segments) > 1:
                positive_transitions = 0
                total_transitions = 0
                
                for i in range(len(strategy_segments) - 1):
                    current_emotion = strategy_segments.iloc[i]['client_emotion']
                    next_emotion = strategy_segments.iloc[i + 1]['client_emotion']
                    
                    if self._is_positive_transition(current_emotion, next_emotion):
                        positive_transitions += 1
                    total_transitions += 1
                
                effectiveness[strategy] = positive_transitions / total_transitions if total_transitions > 0 else 0
                
        return effectiveness

    def _is_positive_transition(self, current_emotion: str, next_emotion: str) -> bool:
        """Determine if an emotion transition is positive."""
        positive_emotions = ['joy', 'neutral']
        negative_emotions = ['anger', 'sadness', 'disgust', 'depression', 'fear']
        
        return (current_emotion in negative_emotions and next_emotion in positive_emotions) or \
               (current_emotion == 'neutral' and next_emotion == 'joy')