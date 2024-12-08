import re
import os
import json
from datetime import timedelta
from google.cloud import storage
from typing import List, Dict, Any
from collections import defaultdict

class TranscriptStatistics:
    """Class to hold and calculate statistics from processed transcripts"""
    def __init__(self):
        self.format_zero_string = "0 (0.00%)"
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
        total_words = sum(self.therapist_words) + sum(self.client_words)
        total_conversations = len(self.therapist_words) + len(self.client_words)
        avg_total_words = total_words / total_conversations if total_conversations > 0 else 0
        
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
                "Total Duration": str(timedelta(milliseconds=self.total_duration_ms)) if self.total_duration_ms > 0 else "0:00:00"
            },
            "Clip Statistics": {
                "Short Clips": f"{self.num_short_clips} ({(self.num_short_clips/self.num_total_clips*100):.2f}%)" if self.num_total_clips > 0 else self.format_zero_string,
                "Short Clips Total Duration": str(timedelta(milliseconds=self.short_clips_duration_ms)) if self.short_clips_duration_ms > 0 else "0:00:00",
                "Average Short Clip Duration": f"{(self.short_clips_duration_ms/1000/self.num_short_clips):.2f}s" if self.num_short_clips > 0 else "0s",
                "Long Clips": f"{self.num_long_clips} ({(self.num_long_clips/self.num_total_clips*100):.2f}%)" if self.num_total_clips > 0 else self.format_zero_string,
                "Long Clips Total Duration": str(timedelta(milliseconds=self.long_clips_duration_ms)) if self.long_clips_duration_ms > 0 else "0:00:00",
                "Average Long Clip Duration": f"{(self.long_clips_duration_ms/1000/self.num_long_clips):.2f}s" if self.num_long_clips > 0 else "0s"
            },
            "Conversation Statistics": {
                "Total Conversations": total_convs,
                "Therapist Conversations": f"{self.num_therapist_convs} ({(self.num_therapist_convs/total_convs*100):.2f}%)" if total_convs > 0 else self.format_zero_string,
                "Client Conversations": f"{self.num_client_convs} ({(self.num_client_convs/total_convs*100):.2f}%)" if total_convs > 0 else self.format_zero_string
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
    def __init__(self, bucket_name: str, project_id: str = None):
        """
        Initialize the TranscriptProcessor with Google Cloud Storage bucket.
        
        Args:
            bucket_name (str): Name of the GCS bucket containing data
            project_id (str, optional): Google Cloud Project ID
        """
        self.bucket_name = bucket_name
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        self.statistics = TranscriptStatistics()
        
        # Print bucket info for debugging
        print(f"Connected to bucket: {self.bucket_name}")
        print(f"Bucket exists: {self.bucket.exists()}")
    
    def count_words(self, text: str) -> int:
        """Count words in a text string"""
        return len(re.findall(r'\w+', text))
    
    def read_json_from_gcs(self, blob_path: str) -> Dict:
        """
        Read JSON file from Google Cloud Storage
        
        Args:
            blob_path (str): Path to the JSON blob in GCS
            
        Returns:
            Dict: Parsed JSON content
            
        Raises:
            FileNotFoundError: If blob doesn't exist
            JSONDecodeError: If content is not valid JSON
        """
        try:
            print(f"Attempting to read JSON from: {blob_path}")
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                raise FileNotFoundError(f"Blob does not exist: {blob_path}")
                
            content = blob.download_as_text()
            return json.loads(content)
        except Exception as e:
            print(f"Error reading JSON from {blob_path}: {str(e)}")
            raise
    
    def list_playlists(self) -> List[str]:
        """
        List all playlist folders in GCS bucket using blob path analysis
        Returns a list of unique playlist prefixes
        """
        try:
            print("\nListing playlists...")
            # List all blobs in the bucket
            all_blobs = list(self.bucket.list_blobs())
            print(f"Total blobs found: {len(all_blobs)}")
            
            # Extract unique playlist prefixes using set comprehension
            playlists = {
                blob.name.split('/')[0] 
                for blob in all_blobs 
                if blob.name.startswith('playlist_')
            }
            
            # Convert to sorted list
            playlist_list = sorted(list(playlists))
            print(f"Found playlists: {playlist_list}")
            return playlist_list
            
        except Exception as e:
            print(f"Error listing playlists: {str(e)}")
            raise

    def list_videos(self, playlist_prefix: str) -> List[str]:
        """
        List all video folders in a playlist using blob path analysis
        
        Args:
            playlist_prefix (str): The playlist prefix to search within
            
        Returns:
            List[str]: List of video folder paths
        """
        try:
            print(f"\nListing videos in playlist: {playlist_prefix}")
            
            # List all blobs with the playlist prefix
            blobs = list(self.bucket.list_blobs(prefix=f"{playlist_prefix}/"))
            
            # Extract unique video paths using set comprehension
            videos = {
                '/'.join(blob.name.split('/')[:2])  # Get playlist_X/video_Y part
                for blob in blobs 
                if 'video_' in blob.name
            }
            
            video_list = sorted(list(videos))
            print(f"Found videos: {video_list}")
            return video_list
            
        except Exception as e:
            print(f"Error listing videos in {playlist_prefix}: {str(e)}")
            raise

    def process_transcript_file(self, gcs_transcript_path: str, video_id: str) -> List[Dict[str, Any]]:
        """
        Process a single transcript file from GCS
        
        Args:
            gcs_transcript_path (str): Path to the transcript JSON file
            video_id (str): Identifier for the video being processed
            
        Returns:
            List[Dict[str, Any]]: List of processed transcript entries
        """
        print(f"\nProcessing transcript: {gcs_transcript_path}")
        transcript_entries = self.read_json_from_gcs(gcs_transcript_path)
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
            
            # Generate GCS path for clip
            clips_dir = os.path.dirname(gcs_transcript_path)
            clip_path = f"{self.bucket_name}/{clips_dir}/clips/clip_{iterator}.mp4"
            clip_path = "gs://" + clip_path.replace('//', '/')
            
            processed_entry = {
                "path": clip_path,
                "role": role,
                "short_clip": is_short,
                "previous_transcript": self.build_previous_transcript(transcript_entries, idx),
                "transcript": entry["text"],
                "emotion": "",
                "strategy": "" if role == "therapist" else "NA",
                "analysis": "NA" if (is_short == "Yes" or role == "therapist") else ""
            }
            processed_data.append(processed_entry)
            
        print(f"Processed {len(processed_data)} entries from transcript")
        return processed_data

    def build_previous_transcript(self, 
                                transcript_entries: List[Dict[Any, Any]], 
                                current_index: int) -> str:
        """
        Build previous transcript string
        
        Args:
            transcript_entries (List[Dict]): List of transcript entries
            current_index (int): Index of current entry
            
        Returns:
            str: Formatted string of previous dialogue
        """
        if current_index == 0:
            return ""
            
        previous_dialogues = []
        for entry in transcript_entries[:current_index]:
            role = "Client" if entry["speaker"] != "Speaker A" else "Therapist"
            previous_dialogues.append(f"{role}: {entry['text']}")
            
        return "\n".join(previous_dialogues)

    def process_all_transcripts(self) -> List[Dict[str, Any]]:
        """
        Process all transcripts in all playlists and videos
        
        Returns:
            List[Dict[str, Any]]: List of all processed transcript entries
        """
        all_processed_data = []
        
        # Get all playlists
        print("\nListing playlists...")
        playlist_prefixes = self.list_playlists()
        if not playlist_prefixes:
            print("No playlists found in bucket!")
            return all_processed_data
            
        self.statistics.num_playlists = len(playlist_prefixes)
        print(f"Found {self.statistics.num_playlists} playlists")
        
        for playlist_prefix in sorted(playlist_prefixes):
            print(f"\nProcessing playlist: {playlist_prefix}")
            # Get all videos in playlist
            video_prefixes = self.list_videos(playlist_prefix)
            if not video_prefixes:
                print(f"No videos found in playlist {playlist_prefix}")
                continue
                
            self.statistics.num_videos += len(video_prefixes)
            print(f"Found {len(video_prefixes)} videos in {playlist_prefix}")
            
            for video_prefix in sorted(video_prefixes):
                print(f"\nProcessing video: {video_prefix}")
                transcript_path = f"{video_prefix}/transcript_timestamps.json"
                
                try:
                    video_data = self.process_transcript_file(transcript_path, f"{playlist_prefix}/{video_prefix}")
                    print(f"Successfully processed {len(video_data)} clips from {transcript_path}")
                    all_processed_data.extend(video_data)
                except Exception as e:
                    print(f"Error processing transcript at {transcript_path}: {str(e)}")
                    continue
        
        self.statistics.num_total_clips = len(all_processed_data)
        print(f"\nTotal processed clips: {self.statistics.num_total_clips}")
        return all_processed_data

    def save_to_json(self, output_path: str):
        """
        Process all transcripts and save results to JSON file
        
        Args:
            output_path (str): Path where the output JSON file should be saved
            
        Raises:
            IOError: If there's an error creating the output directory or writing the file
        """
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        processed_data = self.process_all_transcripts()
        
        if not processed_data:
            print("No data was processed. Nothing to save.")
            return
            
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                print(f"Error creating output directory: {str(e)}")
                raise
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
            print(f"\nSuccessfully processed transcripts and saved to {output_path}")
            
            # Print statistics
            self.statistics.print_statistics()
            
        except Exception as e:
            print(f"Error saving JSON file: {str(e)}")
            raise