"""
This file contains the all of the prompts that are used by the data annotation engine by the Gemini.
"""
import json
from typing import Dict

class AnnotationPrompts:
    """Collection of prompts for different annotation tasks."""
    
    @staticmethod
    def get_emotion_prompt(context: Dict = None) -> str:
        base_prompt = """
        Please analyze the emotional state of the client in this video segment.
        
        Consider the following aspects:
        1. Facial expressions and micro-expressions
        2. Body language and posture
        3. Voice tone, pitch, and rhythm
        4. Verbal content and word choice
        
        Categorize the dominant emotion(s) from:
        - Anger
        - Sadness
        - Disgust
        - Depression
        - Neutral
        - Joy
        - Fear
        
        For each identified emotion, provide:
        1. Confidence score (0-1)
        2. Supporting evidence from the video
        3. Temporal markers for significant changes
        
        Format your response as structured JSON.
        """
        
        if context:
            base_prompt += f"\n\nAdditional context to consider:\n{json.dumps(context, indent=2)}"
        
        return base_prompt

    @staticmethod
    def get_strategy_prompt(context: Dict = None) -> str:
        base_prompt = """
        Analyze the therapeutic strategy employed in this video segment.
        
        Identify the primary therapeutic techniques from:
        - Open questions
        - Approval
        - Self-disclosure
        - Restatement
        - Interpretation
        - Advisement
        - Communication Skills
        - Structuring the therapy
        - Guiding the pace
        
        For each identified strategy:
        1. Provide specific examples from the interaction
        2. Explain the therapeutic intention
        3. Evaluate the immediate impact on the client
        
        Consider:
        - Timing and context of strategy use
        - Client's receptiveness
        - Integration with overall therapeutic goals
        
        Format your response as structured JSON.
        """
        
        if context:
            base_prompt += f"\n\nPrevious session context:\n{json.dumps(context, indent=2)}"
        
        return base_prompt

    @staticmethod
    def get_therapist_emotion_prompt(context: Dict = None) -> str:
        base_prompt = """
        Analyze the therapist's emotional presentation in this video segment.
        
        Evaluate:
        1. Emotional tone and regulation
        2. Level of empathy and attunement
        3. Professional boundaries
        4. Congruence with client's state
        
        Consider:
        - Verbal and non-verbal communication
        - Response timing and pacing
        - Management of countertransference
        - Professional presence
        
        Provide:
        1. Primary emotional states observed
        2. Impact on therapeutic alliance
        3. Effectiveness of emotional containment
        4. Areas for potential improvement
        
        Format your response as structured JSON.
        """
        
        if context:
            base_prompt += f"\n\nRelevant therapeutic history:\n{json.dumps(context, indent=2)}"
        
        return base_prompt

    @staticmethod
    def get_emotion_analysis_prompt(context: Dict = None) -> str:
        base_prompt = """
        Conduct a comprehensive analysis of the emotional dynamics in this therapeutic interaction.
        
        Analyze:
        1. Emotional co-regulation patterns
        2. Attachment dynamics
        3. Emotional safety and trust
        4. Rupture and repair sequences
        
        Consider:
        - Synchrony between client and therapist
        - Emotional depth and authenticity
        - Power dynamics
        - Cultural considerations
        
        Provide:
        1. Key emotional patterns
        2. Critical moments of connection/disconnection
        3. Therapeutic implications
        4. Recommendations for future sessions
        
        Format your response as structured JSON.
        """
        
        if context:
            base_prompt += f"\n\nPrior session dynamics:\n{json.dumps(context, indent=2)}"
        
        return base_prompt