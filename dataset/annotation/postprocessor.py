import re
import json
import random
from logger import setup_logging
from huggingface_hub import create_repo
from datasets import Dataset, DatasetDict
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple, Optional
from constants import (
    CONTEXT_WINDOW,
    TRAIN_RATIO,
    VAL_RATIO,
    RANDOM_SEED,
    CLIP_NUMBER_PATTERN,
    CLIP_PATH_PATTERN,
    INSTRUCTION_TEMPLATE
)

class DatasetConverter:
    """
    A processor for therapy dialogue data that handles grouping, processing,
    and uploading conversation data to Hugging Face datasets.
    
    This class provides functionality to convert raw therapy session data into
    a structured format suitable for machine learning tasks. It handles the
    entire pipeline from data grouping to uploading to Hugging Face.
    
    Attributes:
        context_window (int): Maximum number of previous conversations to include
        train_ratio (float): Proportion of data to use for training
        val_ratio (float): Proportion of data to use for validation
        test_ratio (float): Proportion of data to use for testing
        instruction (str): Template for system instructions
        logger: Logger instance for tracking processing
    """
    
    def __init__(self, 
                 context_window: int = CONTEXT_WINDOW, 
                 train_ratio: float = TRAIN_RATIO,
                 val_ratio: float = VAL_RATIO,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize the DatasetConverter with specified parameters.
        
        Args:
            context_window (int): Number of previous conversations to include
            train_ratio (float): Proportion of data for training
            val_ratio (float): Proportion of data for validation
            random_seed (int): Seed for random operations
        """
        self.context_window = context_window
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.instruction = INSTRUCTION_TEMPLATE
        self.logger = setup_logging(self.__class__.__name__)
        random.seed(random_seed)

    def _check_or_create_repo(self, repo_name: str, token: str) -> bool:
        """
        Check if repository exists, create if it doesn't.
        
        Args:
            repo_name (str): Name of the repository
            token (str): Hugging Face API token
            
        Returns:
            bool: True if repo exists or was created successfully
        """
        try:
            create_repo(repo_name, private=False, token=token, exist_ok=True, repo_type="dataset")
            return True
        except Exception as e:
            self.logger.error(f"Error checking/creating repository: {str(e)}")
            return False

    def format_client_info(self, clip: Dict[str, Any]) -> str:
        """
        Format client information into a structured string.
        
        Args:
            clip (Dict[str, Any]): Client clip data containing transcript and metadata
            
        Returns:
            str: Formatted string containing client information
        """
        parts = [f"Client: {clip['transcript']}"]
        
        if clip['emotion'] not in ['NA', '', ' ']:
            parts.append(f"Client Emotion: {clip['emotion']}")
        
        if clip['analysis'] not in ['NA', '', ' ']:
            parts.append(f"Client Emotion Analysis: {clip['analysis']}")
        
        return '\n'.join(parts)

    def format_therapist_info(self, clip: Dict[str, Any]) -> str:
        """
        Format therapist information into a structured string.
        
        Args:
            clip (Dict[str, Any]): Therapist clip data containing transcript and metadata
            
        Returns:
            str: Formatted string containing therapist information
        """
        parts = [f"Therapist: {clip['transcript']}"]
        
        if clip['emotion'] not in ['NA', '', ' ']:
            parts.append(f"Therapist Emotion: {clip['emotion']}")
        
        if clip['strategy'] not in ['NA', '', ' ']:
            parts.append(f"Therapeutic Strategy: {clip['strategy']}")
        
        return '\n'.join(parts)

    def extract_playlist_video_path(self, path: str) -> Optional[str]:
        """
        Extract playlist and video path from the full path.
        
        Args:
            path (str): Full path to the video clip
            
        Returns:
            Optional[str]: Extracted playlist/video path or None if not found
        """
        match = re.search(CLIP_PATH_PATTERN, path)
        if match:
            return match.group(1)
        return None

    def group_clips_by_video(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group clips by their playlist/video combination.
        
        Args:
            data (List[Dict[str, Any]]): List of clip data to be grouped
            
        Returns:
            Dict[str, Dict[str, Any]]: Clips grouped by video with metadata
        """
        self.logger.info("Starting clip grouping process")
        temp_grouped = defaultdict(list)
        
        for clip in data:
            video_path = self.extract_playlist_video_path(clip['path'])
            if video_path:
                temp_grouped[video_path].append(clip)
        
        result = {}
        for i, (video_path, clips) in enumerate(sorted(temp_grouped.items()), 1):
            new_video_id = f"video_{i}"
            
            sorted_clips = sorted(
                clips,
                key=lambda x: int(re.search(CLIP_NUMBER_PATTERN, x['path']).group(1))
            )
            
            video_info = {
                'clips': sorted_clips,
                'original_path': video_path,
                'total_clips': len(sorted_clips)
            }
            result[new_video_id] = video_info
            self.logger.debug(f"Processed {new_video_id}: {len(sorted_clips)} clips")
        
        self.logger.info(f"Grouped {len(data)} clips into {len(result)} videos")
        return result

    def find_next_therapist_clip(self, clips: List[Dict[str, Any]], start_idx: int) -> Optional[Dict[str, Any]]:
        """
        Find the next therapist clip after the given index.
        
        Args:
            clips (List[Dict[str, Any]]): List of all clips
            start_idx (int): Starting index to search from
            
        Returns:
            Optional[Dict[str, Any]]: Next therapist clip or None if not found
        """
        for i in range(start_idx + 1, len(clips)):
            if clips[i]['role'] == 'therapist':
                return clips[i]
        return None

    def _create_conversation_message(self, 
                                  past_context: str, 
                                  current_clip: Dict[str, Any], 
                                  next_therapist: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create a structured conversation message from clip data.
        
        Args:
            past_context (str): Previous conversation context
            current_clip (Dict[str, Any]): Current client clip
            next_therapist (Dict[str, Any]): Next therapist response
            
        Returns:
            List[Dict[str, str]]: Formatted conversation messages
        """
        return [
            {
                'content': f"{self.instruction}\n\nPast Conversation Context:\n{past_context}",
                'role': 'system'
            },
            {
                'content': ''.join([
                    f"Client Emotion Analysis: {current_clip['analysis']}\n" 
                    if current_clip.get('analysis') not in ['NA', '', ' '] else "",
                    f"{current_clip['transcript']}"
                ]),
                'role': 'user'
            },
            {
                'content': (
                    f"Client Emotion: {current_clip['emotion']}\n"
                    f"Therapeutic Strategy: {next_therapist['strategy']}\n"
                    f"Therapist Emotion: {next_therapist['emotion']}\n"
                    f"Therapist Response: {next_therapist['transcript']}"
                ),
                'role': 'assistant'
            }
        ]

    def process_video_dialogues(self, video_data: Dict[str, Any], start_id: int = 0) -> List[Dict[str, Any]]:
        """
        Process a single video's dialogues with sequential IDs.
        
        Args:
            video_data (Dict[str, Any]): Video data containing clips
            start_id (int): Starting ID for sequential numbering
            
        Returns:
            List[Dict[str, Any]]: List of processed dialogues with IDs
        """
        if not video_data or 'clips' not in video_data:
            self.logger.warning("Invalid video data format")
            return []

        clips = video_data['clips']
        conversations = deque(maxlen=self.context_window)
        processed_dialogues = []
        current_id = start_id

        for i, current_clip in enumerate(clips[:-1]):  # Exclude last clip
            if current_clip['role'] != 'client':
                if current_clip['role'] == 'therapist':
                    conversations.append(self.format_therapist_info(current_clip))
                continue

            next_therapist = self.find_next_therapist_clip(clips, i)
            if not next_therapist:
                continue

            past_context = '\n'.join(list(conversations)) if conversations else "No previous context available."
            
            # Create conversation messages using helper method
            conversation = self._create_conversation_message(
                past_context, current_clip, next_therapist
            )
            
            # Add processed dialogue
            processed_dialogues.append({
                'id': current_id,
                'conversations': conversation
            })
            
            current_id += 1
            conversations.append(self.format_client_info(current_clip))

        return processed_dialogues

    def split_data(self, videos: List[Tuple[str, Dict]]) -> Tuple[List[List[Tuple[str, Dict]]], List[str]]:
        """
        Split videos into train/val/test according to ratios.
        
        Args:
            videos (List[Tuple[str, Dict]]): List of video data to split
            
        Returns:
            Tuple[List[List[Tuple[str, Dict]]], List[str]]: 
                Split data and corresponding split names
        """
        self.logger.info("Starting data split process")
        video_list = list(videos)
        random.shuffle(video_list)
        
        splits = []
        split_names = ['train', 'test'] if self.val_ratio == 0 else ['train', 'val', 'test']
        
        total_videos = len(video_list)
        train_size = int(total_videos * self.train_ratio)
        
        if self.val_ratio > 0:
            val_size = int(total_videos * self.val_ratio)
            splits = [
                video_list[:train_size],
                video_list[train_size:train_size + val_size],
                video_list[train_size + val_size:]
            ]
        else:
            splits = [
                video_list[:train_size],
                video_list[train_size:]
            ]
        
        for name, split in zip(split_names, splits):
            self.logger.info(f"{name} split size: {len(split)} videos")
            
        return splits, split_names

    def process_and_push_to_hf(self, input_file: str, repo_name: str, hf_token: str, commit_message: str) -> None:
        """
        Process the dataset and push directly to Hugging Face.
        
        Args:
            input_file (str): Path to input JSON file
            repo_name (str): Name of the Hugging Face repository
            hf_token (str): Hugging Face API token
            commit_message (str): Commit message for the push
        """
        self.logger.info("Starting data processing pipeline")
        self.logger.info(f"Input file: {input_file}")
        self.logger.info(f"Target repository: {repo_name}")

        # Check/create repository
        if not self._check_or_create_repo(repo_name, hf_token):
            self.logger.error("Failed to access/create repository")
            return

        # Read and group data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        grouped_data = self.group_clips_by_video(data)
        all_videos = list(grouped_data.items())
        
        # Split the data
        splits, split_names = self.split_data(all_videos)
        
        # Process splits and create datasets
        datasets = {}
        current_id = 0
        
        for split_name, videos in zip(split_names, splits):
            all_dialogues = []
            for _, video_data in videos:
                dialogues = self.process_video_dialogues(video_data, current_id)
                all_dialogues.extend(dialogues)
                current_id += len(dialogues)
            
            # Convert to HF Dataset
            dataset = Dataset.from_list(all_dialogues)
            datasets[split_name] = dataset
        
        # Create DatasetDict and push to hub
        dataset_dict = DatasetDict(datasets)
        try:
            dataset_dict.push_to_hub(repo_name, token=hf_token, commit_message=commit_message)
            self.logger.info(f"Successfully pushed dataset to {repo_name}")
        except Exception as e:
            self.logger.error(f"Error pushing to Hugging Face: {str(e)}")