import re
import json
import random
from pathlib import Path
from logger import setup_logging
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
    and splitting of conversation data into training sets.
    
    This class provides functionality to:
    1. Group clips by video
    2. Process dialogues with context windows
    3. Split data into train/val/test sets
    4. Write processed data to JSONL files
    
    Attributes:
        context_window (int): Number of past conversations to maintain in context
        train_ratio (float): Ratio of data to use for training
        val_ratio (float): Ratio of data to use for validation
        test_ratio (float): Ratio of data to use for testing
        instruction (str): Template instruction for the dialogue system
        logger (logging.Logger): Logger instance for the class
    """

    def __init__(self, 
                 context_window: int = CONTEXT_WINDOW, 
                 train_ratio: float = TRAIN_RATIO,
                 val_ratio: float = VAL_RATIO,
                 random_seed: int = RANDOM_SEED):
        """
        Initialize the DialogueProcessor with specified parameters.
        
        Args:
            context_window (int): Number of past conversations to maintain
            train_ratio (float): Ratio of data for training
            val_ratio (float): Ratio of data for validation
            random_seed (int): Random seed for reproducibility
        """
        self.context_window = context_window
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.instruction = INSTRUCTION_TEMPLATE
        self.logger = setup_logging(self.__class__.__name__)
        random.seed(random_seed)

    def format_client_info(self, clip: Dict[str, Any]) -> str:
        """
        Format client information, omitting NA/empty fields.
        
        Args:
            clip (Dict[str, Any]): Client clip data
            
        Returns:
            str: Formatted client information
        """
        parts = [f"Client: {clip['transcript']}"]
        
        if clip['emotion'] not in ['NA', '', ' ']:
            parts.append(f"Client Emotion: {clip['emotion']}")
        
        if clip['analysis'] not in ['NA', '', ' ']:
            parts.append(f"Client Emotion Analysis: {clip['analysis']}")
        
        return '\n'.join(parts)

    def format_therapist_info(self, clip: Dict[str, Any]) -> str:
        """
        Format therapist information, omitting NA/empty fields.
        
        Args:
            clip (Dict[str, Any]): Therapist clip data
            
        Returns:
            str: Formatted therapist information
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
            path (str): Full path to the clip
            
        Returns:
            Optional[str]: Extracted path or None if no match found
        
        Example:
            >>> processor.extract_playlist_video_path('../../data/playlist_1/video_1/clips/clip_002.mp4')
            'playlist_1/video_1'
        """
        match = re.search(CLIP_PATH_PATTERN, path)
        if match:
            return match.group(1)
        return None

    def group_clips_by_video(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Group clips by their playlist/video combination.
        
        Args:
            data (List[Dict[str, Any]]): List of clip data
            
        Returns:
            Dict[str, Dict[str, Any]]: Grouped clips with video IDs as keys
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
            clips (List[Dict[str, Any]]): List of clips
            start_idx (int): Starting index to search from
            
        Returns:
            Optional[Dict[str, Any]]: Next therapist clip or None if not found
        """
        for i in range(start_idx + 1, len(clips)):
            if clips[i]['role'] == 'therapist':
                return clips[i]
        return None

    def process_video_dialogues(self, video_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Process a single video's dialogues.
        
        Args:
            video_data (Dict[str, Any]): Video data containing clips
            
        Returns:
            List[Dict[str, str]]: List of processed dialogues
        """
        if not video_data or 'clips' not in video_data:
            self.logger.warning("Invalid video data format")
            return []

        clips = video_data['clips']
        conversations = deque(maxlen=self.context_window)
        processed_dialogues = []

        for i, current_clip in enumerate(clips):
            if current_clip['role'] != 'client':
                if current_clip['role'] == 'therapist':
                    conversations.append(self.format_therapist_info(current_clip))
                continue

            if i == len(clips) - 1:
                continue

            next_therapist = self.find_next_therapist_clip(clips, i)
            if not next_therapist:
                continue

            past_context = '\n'.join(list(conversations)) if conversations else "No previous context available."

            # First, create the instruction with context
            full_instruction = (
                f"{self.instruction}\n\n"
                f"Past Conversation Context:\n{past_context}"
            )

            # Build input string conditionally including emotion analysis
            input_parts = []
            # Add emotion analysis if it exists and is meaningful
            if current_clip.get('analysis') not in ['NA', '', ' ']:
                input_parts.append(f"Client Emotion Analysis: {current_clip['analysis']}\n")

            input_parts.append(f"Client: {current_clip['transcript']}")

            # Create the dialogue dictionary
            dialogue = {
                'instruction': full_instruction,
                'input': ''.join(input_parts),
                'output': (
                    f"Client Emotion: {current_clip['emotion']}\n"
                    f"Therapeutic Strategy: {next_therapist['strategy']}\n"
                    f"Therapist Emotion: {next_therapist['emotion']}\n"
                    f"Therapist: {next_therapist['transcript']}"
                )
            }
                        
            processed_dialogues.append(dialogue)
            conversations.append(self.format_client_info(current_clip))

        return processed_dialogues

    def split_data(self, videos: List[Tuple[str, Dict]]) -> Tuple[List[List[Tuple[str, Dict]]], List[str]]:
        """
        Split videos into train/val/test according to ratios.
        
        Args:
            videos (List[Tuple[str, Dict]]): List of video data tuples
            
        Returns:
            Tuple[List[List[Tuple[str, Dict]]], List[str]]: Split data and split names
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

    def write_jsonl(self, data: List[Dict], output_path: Path):
        """
        Write dialogues to a JSONL file.
        
        Args:
            data (List[Dict]): Data to write
            output_path (Path): Path to output file
        """
        self.logger.info(f"Writing data to {output_path}")
        with output_path.open('w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        self.logger.info(f"Wrote {len(data)} entries to {output_path}")

    def process_data(self, input_file: str, output_dir: str) -> None:
        """
        Process the entire dataset and generate JSONL files.
        
        Args:
            input_file (str): Path to input JSON file
            output_dir (str): Directory for output files
        """
        self.logger.info("Starting data processing pipeline")
        self.logger.info(f"Input file: {input_file}")
        self.logger.info(f"Output directory: {output_dir}")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read and group data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        grouped_data = self.group_clips_by_video(data)
        all_videos = list(grouped_data.items())
        
        # Split the data
        splits, split_names = self.split_data(all_videos)
        
        # Process and write each split
        for split_name, videos in zip(split_names, splits):
            output_path = output_dir / f'{split_name}.jsonl'
            
            all_dialogues = []
            for _, video_data in videos:
                dialogues = self.process_video_dialogues(video_data)
                all_dialogues.extend(dialogues)
            
            self.write_jsonl(all_dialogues, output_path)