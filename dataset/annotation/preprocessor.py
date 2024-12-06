"""
This file contains the code related to the pre-processing of the videos.
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import librosa

class VideoPreprocessor:
    def __init__(self, config: Dict):
        self.frame_rate = config.get('frame_rate', 30)
        self.audio_sample_rate = config.get('audio_sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration', 10)  # seconds
        
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video at specified frame rate."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        return frames

    def extract_audio(self, video_path: str) -> np.ndarray:
        """Extract audio from video and resample."""
        audio, _ = librosa.load(video_path, sr=self.audio_sample_rate)
        return audio

    def segment_video(self, video_path: str) -> List[Dict]:
        """Segment video into chunks with metadata."""
        frames = self.extract_frames(video_path)
        audio = self.extract_audio(video_path)
        
        chunks = []
        total_frames = len(frames)
        frames_per_chunk = self.frame_rate * self.chunk_duration
        
        for i in range(0, total_frames, frames_per_chunk):
            chunk_frames = frames[i:i + frames_per_chunk]
            chunk_start_time = i / self.frame_rate
            
            chunk_data = {
                'start_time': chunk_start_time,
                'duration': self.chunk_duration,
                'frames': chunk_frames,
                'audio': audio[int(chunk_start_time * self.audio_sample_rate):
                            int((chunk_start_time + self.chunk_duration) * self.audio_sample_rate)]
            }
            chunks.append(chunk_data)
            
        return chunks