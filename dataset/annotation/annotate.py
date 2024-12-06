"""
This file contains the code for the data annotation engine.
"""

import cv2
import base64
import asyncio
import vertexai
from typing import Dict
from vertexai.generative_models import GenerativeModel


class TherapyAnnotator:
    def __init__(self, project_id: str, location: str):
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-pro")
        
        self.emotion_prompt = """
        Analyze the following video segment for client emotions. Consider:
        - Facial expressions
        - Body language
        - Voice tone and intonation
        - Verbal content
        
        Categorize the dominant emotions from: {anger, sadness, disgust, depression, neutral, joy, fear}
        Provide confidence scores and supporting evidence.
        """
        
        self.strategy_prompt = """
        Based on the therapeutic interaction, identify the therapist's strategy:
        - Open questions
        - Approval
        - Self-disclosure
        - Restatement
        - Interpretation
        - Advisement
        - Communication Skills
        - Structuring the therapy
        - Guiding the pace
        
        Explain the reasoning behind the strategy identification.
        """
        
        self.therapist_emotion_prompt = """
        Analyze the therapist's emotional response in this segment:
        - Emotional tone
        - Level of empathy
        - Professional boundaries
        - Alignment with client's state
        
        Provide specific examples supporting the analysis.
        """

    def _prepare_video_input(self, chunk: Dict) -> str:
        """Convert video chunk to base64 for API input."""
        frames = chunk['frames']
        # Encode frames as video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_path = 'temp_chunk.mp4'
        out = cv2.VideoWriter(temp_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Convert to base64
        with open(temp_path, 'rb') as f:
            video_bytes = f.read()
        return base64.b64encode(video_bytes).decode()

    async def annotate_chunk(self, chunk: Dict) -> Dict:
        """Annotate a single video chunk with all analysis tasks."""
        video_input = self._prepare_video_input(chunk)
        
        # Parallel annotation tasks
        tasks = [
            self._annotate_emotion(video_input),
            self._annotate_strategy(video_input),
            self._annotate_therapist_emotion(video_input)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            'timestamp': chunk['start_time'],
            'duration': chunk['duration'],
            'client_emotion': results[0],
            'therapist_strategy': results[1],
            'therapist_emotion': results[2]
        }

    async def _annotate_emotion(self, video_input: str) -> Dict:
        """Analyze client emotions using Gemini."""
        response = await self.model.generate_content(
            self.emotion_prompt,
            video_input,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        return self._parse_emotion_response(response.text)

    async def _annotate_strategy(self, video_input: str) -> Dict:
        """Analyze therapist strategy using Gemini."""
        response = await self.model.generate_content(
            self.strategy_prompt,
            video_input,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.9
            }
        )
        return self._parse_strategy_response(response.text)

    async def _annotate_therapist_emotion(self, video_input: str) -> Dict:
        """Analyze therapist emotion using Gemini."""
        response = await self.model.generate_content(
            self.therapist_emotion_prompt,
            video_input,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8
            }
        )
        return self._parse_therapist_emotion_response(response.text)