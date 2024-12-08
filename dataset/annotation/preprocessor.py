import os
import re
import json
from datetime import timedelta
from typing import List, Dict, Any
from collections import defaultdict

class TranscriptStatistics:
    """Class to hold and calculate statistics from processed transcripts"""
    def __init__(self):
        self.num_playlists = 0
        self.num_videos = 0
        self.num_total_clips = 0
        self.num_short_clips = 0
        self.num_long_clips = 0
        self.num_therapist_convs = 0
        self.num_client_convs = 0
        self.therapist_words = []
        self.client_words = []
        self.therapist_turns_per_video = defaultdict(int)
        self.client_turns_per_video = defaultdict(int)
        self.total_duration_ms = 0
        self.short_clips_duration_ms = 0
        self.long_clips_duration_ms = 0
        
    def calculate_word_stats(self) -> Dict[str, float]:
        """Calculate word statistics for conversations"""
        avg_therapist_words = sum(self.therapist_words) / len(self.therapist_words) if self.therapist_words else 0
        avg_client_words = sum(self.client_words) / len(self.client_words) if self.client_words else 0
        avg_total_words = (sum(self.therapist_words) + sum(self.client_words)) / (
            len(self.therapist_words) + len(self.client_words)) if (self.therapist_words or self.client_words) else 0
        
        return {
            "avg_therapist_words": round(avg_therapist_words, 2),
            "avg_client_words": round(avg_client_words, 2),
            "avg_total_words": round(avg_total_words, 2)
        }
    
    def calculate_turn_stats(self) -> Dict[str, float]:
        """Calculate average turns per video"""
        avg_therapist_turns = sum(self.therapist_turns_per_video.values()) / len(self.therapist_turns_per_video) if self.therapist_turns_per_video else 0
        avg_client_turns = sum(self.client_turns_per_video.values()) / len(self.client_turns_per_video) if self.client_turns_per_video else 0
        
        return {
            "avg_therapist_turns": round(avg_therapist_turns, 2),
            "avg_client_turns": round(avg_client_turns, 2)
        }
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        word_stats = self.calculate_word_stats()
        turn_stats = self.calculate_turn_stats()
        total_convs = self.num_therapist_convs + self.num_client_convs
        
        stats = {
            "Dataset Overview": {
                "Number of Playlists": self.num_playlists,
                "Number of Videos": self.num_videos,
                "Total Number of Clips": self.num_total_clips,
                "Total Duration": str(timedelta(milliseconds=self.total_duration_ms))
            },
            "Clip Statistics": {
                "Short Clips": f"{self.num_short_clips} ({(self.num_short_clips/self.num_total_clips*100):.2f}%)",
                "Short Clips Total Duration": str(timedelta(milliseconds=self.short_clips_duration_ms)),
                "Average Short Clip Duration": f"{(self.short_clips_duration_ms/1000/self.num_short_clips):.2f}s" if self.num_short_clips > 0 else "0s",
                "Long Clips": f"{self.num_long_clips} ({(self.num_long_clips/self.num_total_clips*100):.2f}%)",
                "Long Clips Total Duration": str(timedelta(milliseconds=self.long_clips_duration_ms)),
                "Average Long Clip Duration": f"{(self.long_clips_duration_ms/1000/self.num_long_clips):.2f}s" if self.num_long_clips > 0 else "0s"
            },
            "Conversation Statistics": {
                "Total Conversations": total_convs,
                "Therapist Conversations": f"{self.num_therapist_convs} ({(self.num_therapist_convs/total_convs*100):.2f}%)",
                "Client Conversations": f"{self.num_client_convs} ({(self.num_client_convs/total_convs*100):.2f}%)"
            },
            "Turn Taking Analysis": {
                "Average Therapist Turns per Video": turn_stats["avg_therapist_turns"],
                "Average Client Turns per Video": turn_stats["avg_client_turns"],
                "Total Turns per Video": f"{turn_stats['avg_therapist_turns'] + turn_stats['avg_client_turns']:.2f}"
            },
            "Word Count Analysis": {
                "Average Words per Therapist Turn": word_stats["avg_therapist_words"],
                "Average Words per Client Turn": word_stats["avg_client_words"],
                "Average Words per Turn (Overall)": word_stats["avg_total_words"]
            }
        }
        
        print("\n=== Dataset Statistics ===")
        for category, category_stats in stats.items():
            print(f"\n{category}:")
            for key, value in category_stats.items():
                print(f"  {key}: {value}")

class TranscriptProcessor:
    def __init__(self, base_path: str):
        """
        Initialize the TranscriptProcessor with the base path containing playlists.
        
        Args:
            base_path (str): Base directory path containing playlist folders
        """
        self.base_path = os.path.normpath(base_path)
        self.statistics = TranscriptStatistics()
        
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Directory not found: {self.base_path}")
    
    def count_words(self, text: str) -> int:
        """Count words in a text string"""
        return len(re.findall(r'\w+', text))
    
    def process_transcript_file(self, transcript_path: str, video_id: str) -> List[Dict[str, Any]]:
        """Process a single transcript file"""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_entries = json.load(f)
            
        processed_data = []
        
        for idx, entry in enumerate(transcript_entries):
            role = "therapist" if entry["speaker"] == "Speaker A" else "client"
            iterator = f"{idx + 1:03d}"
            
            # Update statistics
            if role == "therapist":
                self.statistics.num_therapist_convs += 1
                self.statistics.therapist_words.append(self.count_words(entry["text"]))
                self.statistics.therapist_turns_per_video[video_id] += 1
            else:
                self.statistics.num_client_convs += 1
                self.statistics.client_words.append(self.count_words(entry["text"]))
                self.statistics.client_turns_per_video[video_id] += 1
            
            # Calculate duration and update statistics
            duration = entry["end"] - entry["start"]
            self.statistics.total_duration_ms += duration
            is_short = "Yes" if duration/1000 < 3 else "No"
            if is_short == "Yes":
                self.statistics.num_short_clips += 1
                self.statistics.short_clips_duration_ms += duration
            else:
                self.statistics.num_long_clips += 1
                self.statistics.long_clips_duration_ms += duration
            
            processed_entry = {
                "path": os.path.join(os.path.dirname(transcript_path), "clips", f"clip_{iterator}.mp4"),
                "role": role,
                "short_clip": is_short,
                "previous_transcript": self.build_previous_transcript(transcript_entries, idx),
                "transcript": entry["text"],
                "emotion": "",
                "strategy": "" if role == "therapist" else "NA",
                "analysis": "" if (is_short == "No" and role == "client") else "NA"
            }
            processed_data.append(processed_entry)
            
        return processed_data

    def build_previous_transcript(self, 
                                transcript_entries: List[Dict[Any, Any]], 
                                current_index: int) -> str:
        """Build previous transcript string"""
        if current_index == 0:
            return ""
            
        previous_dialogues = []
        for entry in transcript_entries[:current_index]:
            role = "Client" if entry["speaker"] != "Speaker A" else "Therapist"
            previous_dialogues.append(f"{role}: {entry['text']}")
            
        return "\n".join(previous_dialogues)

    def process_all_transcripts(self) -> List[Dict[str, Any]]:
        """Process all transcripts in all playlists and videos"""
        all_processed_data = []
        
        # Iterate through playlist directories (only folders)
        playlist_dirs = [d for d in os.listdir(self.base_path) 
                        if d.startswith("playlist_") and os.path.isdir(os.path.join(self.base_path, d))]
        self.statistics.num_playlists = len(playlist_dirs)
        
        for playlist_dir in sorted(playlist_dirs):
            playlist_path = os.path.join(self.base_path, playlist_dir)
            
            # Iterate through video directories (only folders)
            video_dirs = [d for d in os.listdir(playlist_path) 
                         if d.startswith("video_") and os.path.isdir(os.path.join(playlist_path, d))]
            self.statistics.num_videos += len(video_dirs)
            
            for video_dir in sorted(video_dirs):
                video_path = os.path.join(playlist_path, video_dir)
                transcript_path = os.path.join(video_path, "transcript_timestamps.json")
                
                if os.path.exists(transcript_path):
                    video_data = self.process_transcript_file(transcript_path, f"{playlist_dir}/{video_dir}")
                    all_processed_data.extend(video_data)
                else:
                    print(f"Warning: No transcript file found in {video_path}")
        
        self.statistics.num_total_clips = len(all_processed_data)
        return all_processed_data

    def save_to_json(self, output_path: str):
        """Process all transcripts and save results to JSON file"""
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        processed_data = self.process_all_transcripts()
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
            print(f"\nSuccessfully processed transcripts and saved to {output_path}")
            
            # Print statistics
            self.statistics.print_statistics()
            
        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")