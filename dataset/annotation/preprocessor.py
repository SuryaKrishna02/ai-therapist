import re
import os
import json
import shutil
import logging
import colorlog
import subprocess
from pathlib import Path
from datetime import timedelta
from google.cloud import storage
from collections import defaultdict
from typing import List, Dict, Any, Tuple

def setup_logging(name: str) -> logging.Logger:
    """
    Set up logging configuration with colorlog
    
    Args:
        name (str): Logger name to be used for the logging instance
        
    Returns:
        logging.Logger: Configured logger instance with color formatting
    
    Example:
        logger = setup_logging("my_module")
        logger.info("This will be displayed in green")
        logger.error("This will be displayed in red")
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s%(reset)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    ))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logs
    for existing_handler in logger.handlers[:]:
        logger.removeHandler(existing_handler)
    
    logger.addHandler(handler)
    return logger

class TranscriptStatistics:
    """
    A class to hold and calculate statistics from processed transcripts.
    
    This class maintains counters and collections for various statistics about transcripts,
    including counts of playlists, videos, clips, conversations, and various metrics
    about clip durations and corruption status. It provides methods to calculate and
    format these statistics for presentation.
    
    Attributes:
        format_zero_string (str): Default format for zero values in statistics
        num_playlists (int): Total number of playlists processed
        num_videos (int): Total number of videos processed
        num_total_clips (int): Total number of clips processed
        num_short_clips (int): Number of clips shorter than threshold
        num_long_clips (int): Number of clips longer than threshold
        num_corrupted_clips (int): Number of corrupted clips
        num_normal_clips (int): Number of clips that are neither corrupted nor short
        num_therapist_convs (int): Number of therapist conversations
        num_client_convs (int): Number of client conversations
        therapist_words (List[int]): Word counts for therapist utterances
        client_words (List[int]): Word counts for client utterances
        therapist_turns_per_video (DefaultDict[str, int]): Count of therapist turns by video
        client_turns_per_video (DefaultDict[str, int]): Count of client turns by video
        total_duration_ms (int): Total duration of all clips in milliseconds
        short_clips_duration_ms (int): Total duration of short clips
        long_clips_duration_ms (int): Total duration of long clips
        corrupted_clips_duration_ms (int): Total duration of corrupted clips
        normal_clips_duration_ms (int): Total duration of normal clips
        corruption_reasons (DefaultDict[str, int]): Counts of corruption reasons
    """
    
    def __init__(self):
        """
        Initialize statistics tracking with default values for all counters and collections.
        
        Sets up the logging configuration and initializes all statistics counters and
        collections to their default values.
        """
        self.logger = setup_logging(f"{__name__}.TranscriptStatistics")
        self.format_zero_string = "0 (0.00%)"
        self.num_playlists = 0
        self.num_videos = 0
        self.num_total_clips = 0
        self.num_short_clips = 0
        self.num_long_clips = 0
        self.num_corrupted_clips = 0
        self.num_normal_clips = 0
        self.num_therapist_convs = 0
        self.num_client_convs = 0
        self.therapist_words = []
        self.client_words = []
        self.therapist_turns_per_video = defaultdict(int)
        self.client_turns_per_video = defaultdict(int)
        self.total_duration_ms = 0
        self.short_clips_duration_ms = 0
        self.long_clips_duration_ms = 0
        self.corrupted_clips_duration_ms = 0
        self.normal_clips_duration_ms = 0
        self.corruption_reasons = defaultdict(int)

    def calculate_word_stats(self) -> Dict[str, float]:
        """
        Calculate word statistics for conversations.
        
        Computes average word counts for therapist and client turns, as well as
        overall average words per turn.
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - avg_therapist_words: Average words per therapist turn
                - avg_client_words: Average words per client turn
                - avg_total_words: Average words per turn overall
        """
        avg_therapist_words = sum(self.therapist_words) / len(self.therapist_words) if self.therapist_words else 0
        avg_client_words = sum(self.client_words) / len(self.client_words) if self.client_words else 0
        total_words = sum(self.therapist_words) + sum(self.client_words)
        total_conversations = len(self.therapist_words) + len(self.client_words)
        avg_total_words = total_words / total_conversations if total_conversations > 0 else 0
        
        self.logger.debug("Calculated word statistics")
        return {
            "avg_therapist_words": round(avg_therapist_words, 2),
            "avg_client_words": round(avg_client_words, 2),
            "avg_total_words": round(avg_total_words, 2)
        }

    def calculate_turn_stats(self) -> Dict[str, float]:
        """
        Calculate average turns per video statistics.
        
        Computes the average number of turns per video for both therapist
        and client interactions.
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - avg_therapist_turns: Average therapist turns per video
                - avg_client_turns: Average client turns per video
        """
        avg_therapist_turns = sum(self.therapist_turns_per_video.values()) / len(self.therapist_turns_per_video) if self.therapist_turns_per_video else 0
        avg_client_turns = sum(self.client_turns_per_video.values()) / len(self.client_turns_per_video) if self.client_turns_per_video else 0
        
        self.logger.debug("Calculated turn statistics")
        return {
            "avg_therapist_turns": round(avg_therapist_turns, 2),
            "avg_client_turns": round(avg_client_turns, 2)
        }

    def _format_percentage(self, count: int, total: int) -> str:
        """
        Format count as percentage of total.
        
        Args:
            count (int): Number to calculate percentage for
            total (int): Total number for percentage calculation
            
        Returns:
            str: Formatted string with count and percentage
            
        Example:
            >>> _format_percentage(25, 100)
            '25 (25.00%)'
        """
        if total > 0:
            percentage = (count / total * 100)
            return f"{count} ({percentage:.2f}%)"
        return self.format_zero_string

    def _format_duration(self, duration_ms: int, count: int = None) -> Dict[str, str]:
        """
        Format duration in milliseconds to human readable format.
        
        Args:
            duration_ms (int): Duration in milliseconds
            count (int, optional): Number of items for average calculation
            
        Returns:
            Dict[str, str]: Dictionary containing:
                - total: Total duration in HH:MM:SS format
                - average: Average duration in seconds (if count provided)
                
        Example:
            >>> _format_duration(3600000, 10)
            {'total': '1:00:00', 'average': '360.00s'}
        """
        result = {
            "total": str(timedelta(milliseconds=duration_ms)) if duration_ms > 0 else "0:00:00"
        }
        
        if count is not None and count > 0:
            avg_duration = duration_ms / 1000 / count
            result["average"] = f"{avg_duration:.2f}s"
        else:
            result["average"] = "0s"
            
        return result

    def _get_clip_statistics(self) -> Dict[str, Any]:
        """
        Generate clip statistics dictionary.
        
        Calculates and formats statistics about different types of clips
        including short clips, long clips, and normal clips.
        
        Returns:
            Dict[str, Any]: Dictionary containing formatted statistics about:
                - Short clips count, duration, and average duration
                - Long clips count, duration, and average duration
                - Normal clips count, duration, and average duration
        """
        short_durations = self._format_duration(self.short_clips_duration_ms, self.num_short_clips)
        long_durations = self._format_duration(self.long_clips_duration_ms, self.num_long_clips)
        normal_durations = self._format_duration(self.normal_clips_duration_ms, self.num_normal_clips)
        
        return {
            "Short Clips": self._format_percentage(self.num_short_clips, self.num_total_clips),
            "Short Clips Total Duration": short_durations["total"],
            "Average Short Clip Duration": short_durations["average"],
            "Long Clips": self._format_percentage(self.num_long_clips, self.num_total_clips),
            "Long Clips Total Duration": long_durations["total"],
            "Average Long Clip Duration": long_durations["average"],
            "Normal Clips (Not corrupted, Not short)": self._format_percentage(self.num_normal_clips, self.num_total_clips),
            "Normal Clips Total Duration": normal_durations["total"],
            "Average Normal Clip Duration": normal_durations["average"]
        }

    def _get_corruption_statistics(self) -> Dict[str, Any]:
        """
        Generate corruption statistics dictionary.
        
        Calculates and formats statistics about corrupted and non-corrupted clips
        including counts, durations, and reasons for corruption.
        
        Returns:
            Dict[str, Any]: Dictionary containing formatted statistics about:
                - Corrupted clips count, duration, and average duration
                - Non-corrupted clips count
                - Reasons for corruption and their frequencies
        """
        corrupted_durations = self._format_duration(self.corrupted_clips_duration_ms, self.num_corrupted_clips)
        non_corrupted_clips = self.num_total_clips - self.num_corrupted_clips
        
        return {
            "Corrupted Clips": self._format_percentage(self.num_corrupted_clips, self.num_total_clips),
            "Corrupted Clips Total Duration": corrupted_durations["total"],
            "Average Corrupted Clip Duration": corrupted_durations["average"],
            "Non-corrupted Clips": self._format_percentage(non_corrupted_clips, self.num_total_clips),
            "Corruption Reasons": dict(self.corruption_reasons)
        }

    def _get_conversation_statistics(self, total_convs: int) -> Dict[str, str]:
        """
        Generate conversation statistics dictionary.
        
        Args:
            total_convs (int): Total number of conversations
            
        Returns:
            Dict[str, str]: Dictionary containing:
                - Total conversation count
                - Therapist conversations count and percentage
                - Client conversations count and percentage
        """
        return {
            "Total Conversations": total_convs,
            "Therapist Conversations": self._format_percentage(self.num_therapist_convs, total_convs),
            "Client Conversations": self._format_percentage(self.num_client_convs, total_convs)
        }

    def print_statistics(self):
        """
        Print comprehensive statistics about the processed transcripts.
        
        This method compiles and displays various statistics about the processed
        dataset, including clip counts, durations, corruption rates, and conversation
        metrics. It organizes the statistics into categories and formats them for
        clear presentation.
        
        The statistics are displayed in the following categories:
            - Dataset Overview
            - Clip Statistics
            - Corruption Statistics
            - Conversation Statistics
            - Turn Taking Analysis
            - Word Count Analysis
            
        Each category is logged using the configured logger with appropriate
        formatting and organization.
        """
        self.logger.info("Generating comprehensive statistics report")
        word_stats = self.calculate_word_stats()
        turn_stats = self.calculate_turn_stats()
        total_convs = self.num_therapist_convs + self.num_client_convs
        
        # Compile all statistics
        stats = {
            "Dataset Overview": {
                "Number of Playlists": self.num_playlists,
                "Number of Videos": self.num_videos,
                "Total Number of Clips": self.num_total_clips,
                "Total Duration": self._format_duration(self.total_duration_ms)["total"]
            },
            "Clip Statistics": self._get_clip_statistics(),
            "Corruption Statistics": self._get_corruption_statistics(),
            "Conversation Statistics": self._get_conversation_statistics(total_convs),
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
        
        # Display statistics
        self.logger.info("\n=== Dataset Statistics ===")
        for category, category_stats in stats.items():
            self.logger.info(f"\n{category}:")
            for key, value in category_stats.items():
                self.logger.info(f"  {key}: {value}")

class TranscriptProcessor:
    """
    A class for processing transcript files from Google Cloud Storage.
    
    This class handles the extraction, analysis, and processing of transcript
    files stored in Google Cloud Storage. It manages temporary file storage,
    corruption checking, and statistical analysis of the processed transcripts.
    
    Attributes:
        bucket_name (str): Name of the GCS bucket containing data
        storage_client (storage.Client): Google Cloud Storage client
        bucket (storage.Bucket): GCS bucket instance
        statistics (TranscriptStatistics): Statistics tracking instance
        temp_dir (Path): Path to temporary file storage directory
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(self, bucket_name: str, project_id: str = None):
        """
        Initialize the TranscriptProcessor with Google Cloud Storage bucket.
        
        Args:
            bucket_name (str): Name of the GCS bucket containing data
            project_id (str, optional): Google Cloud Project ID. Defaults to None.
        
        Raises:
            google.cloud.exceptions.NotFound: If bucket doesn't exist
            google.auth.exceptions.DefaultCredentialsError: If GCP credentials are invalid
        """
        self.logger = setup_logging(f"{__name__}.TranscriptProcessor")
        
        # Set up temporary directory
        self.temp_dir = Path("tmp")
        self.temp_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created temporary directory at {self.temp_dir}")
        
        self.bucket_name = bucket_name
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        self.statistics = TranscriptStatistics()
        
        self.logger.info(f"Connected to bucket: {self.bucket_name}")
        self.logger.info(f"Bucket exists: {self.bucket.exists()}")
    
    def __del__(self):
        """
        Cleanup temporary files when the object is destroyed.
        
        Note:
            Uses print instead of logger during cleanup to avoid issues
            during interpreter shutdown when logging might not be available.
        """
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {str(e)}")
    
    def count_words(self, text: str) -> int:
        """
        Count words in a text string.
        
        Args:
            text (str): The text to count words in
            
        Returns:
            int: Number of words in the text
            
        Note:
            Words are defined as sequences of word characters (\w+)
        """
        return len(re.findall(r'\w+', text))
    
    def read_json_from_gcs(self, blob_path: str) -> Dict:
        """
        Read JSON file from Google Cloud Storage.
        
        Args:
            blob_path (str): Path to the JSON blob in GCS
            
        Returns:
            Dict: Parsed JSON content
            
        Raises:
            FileNotFoundError: If blob doesn't exist
            JSONDecodeError: If content is not valid JSON
        """
        try:
            self.logger.info(f"Reading JSON from: {blob_path}")
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                raise FileNotFoundError(f"Blob does not exist: {blob_path}")
                
            content = blob.download_as_text()
            return json.loads(content)
        except Exception as e:
            self.logger.error(f"Error reading JSON from {blob_path}: {str(e)}")
            raise

    def check_video_corruption(self, clip_path: str) -> Tuple[bool, str]:
        """
        Check if a video clip is corrupted by examining its video streams using ffprobe.
        
        Args:
            clip_path (str): GCS path to the video clip
            
        Returns:
            Tuple[bool, str]: 
                - First element: True if corrupted/audio-only, False if valid video
                - Second element: Description of the issue if corrupted
                
        Raises:
            subprocess.SubprocessError: If ffprobe command fails
            JSONDecodeError: If ffprobe output cannot be parsed
        """
        try:
            clean_path = clip_path.replace(f"gs://{self.bucket_name}/", "")
            blob = self.bucket.blob(clean_path)
            
            if not blob.exists():
                return False, "File not found"
            
            # Create a temporary file in our tmp directory
            temp_file_path = self.temp_dir / f"temp_clip_{hash(clip_path)}.mp4"
            self.logger.debug(f"Downloading clip to temporary file: {temp_file_path}")
            
            blob.download_to_filename(str(temp_file_path))
            
            # Use ffprobe to get stream information
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-show_format',
                str(temp_file_path)
            ]
            
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            
            # Clean up the temporary file
            temp_file_path.unlink()
            
            if result.returncode != 0:
                self.logger.error(f"FFprobe error for {clip_path}: {result.stderr}")
                return True, f"FFprobe error: {result.stderr}"
            
            try:
                probe_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse FFprobe output for {clip_path}")
                return True, "Failed to parse FFprobe output"
            
            # Check for presence of streams
            streams = probe_data.get('streams', [])
            has_video = False
            has_audio = False
            video_info = {}
            
            for stream in streams:
                codec_type = stream.get('codec_type')
                if codec_type == 'video':
                    has_video = True
                    video_info = {
                        'fps': stream.get('r_frame_rate', 'N/A'),
                        'width': stream.get('width', 0),
                        'height': stream.get('height', 0),
                        'codec': stream.get('codec_name', 'unknown')
                    }
                elif codec_type == 'audio':
                    has_audio = True
            
            # Check corruption scenarios
            if not has_video:
                return True, "No video stream found (audio-only file)"
            
            if not has_audio:
                return True, "No audio stream found (video-only file)"
            
            if video_info.get('width', 0) <= 0 or video_info.get('height', 0) <= 0:
                return True, "Invalid video dimensions"
            
            # Check framerate
            fps_str = video_info.get('fps', '0/1')
            try:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 0
                if fps <= 0:
                    return True, f"Invalid framerate: {fps_str}"
            except (ValueError, ZeroDivisionError):
                return True, f"Could not parse framerate: {fps_str}"
            
            return False, "Valid video file"
            
        except Exception as e:
            self.logger.error(f"Error checking video corruption for {clip_path}: {str(e)}")
            return True, f"Error checking video: {str(e)}"

    def list_playlists(self) -> List[str]:
        """
        List all playlist folders in GCS bucket using blob path analysis.
        
        Returns:
            List[str]: List of unique playlist prefixes
            
        Raises:
            google.cloud.exceptions.NotFound: If bucket doesn't exist
            google.api_core.exceptions.GoogleAPIError: For GCS API errors
        """
        try:
            self.logger.info("Listing playlists...")
            all_blobs = list(self.bucket.list_blobs())
            self.logger.info(f"Total blobs found: {len(all_blobs)}")
            
            playlists = {
                blob.name.split('/')[0] 
                for blob in all_blobs 
                if blob.name.startswith('playlist_')
            }
            
            playlist_list = sorted(list(playlists))
            self.logger.info(f"Found playlists: {playlist_list}")
            return playlist_list
            
        except Exception as e:
            self.logger.error(f"Error listing playlists: {str(e)}")
            raise

    def list_videos(self, playlist_prefix: str) -> List[str]:
        """
        List all video folders in a playlist that match the 'video_X' pattern.
        
        Args:
            playlist_prefix (str): The playlist prefix to search within (e.g., 'playlist_1')
            
        Returns:
            List[str]: List of video folder paths matching the pattern
            
        Raises:
            google.cloud.exceptions.NotFound: If playlist doesn't exist
            google.api_core.exceptions.GoogleAPIError: For GCS API errors
            
        Note:
            Only looks for folders matching pattern 'video_X' where X is a number,
            ignoring any other files or folders in the playlist directory.
        """
        try:
            self.logger.info(f"Listing videos in playlist: {playlist_prefix}")
            
            # List all blobs with the playlist prefix
            blobs = list(self.bucket.list_blobs(prefix=f"{playlist_prefix}/"))
            
            # Extract unique video folder paths using regex pattern
            videos = set()
            video_pattern = re.compile(rf"{playlist_prefix}/video_\d+/")
            
            for blob in blobs:
                match = video_pattern.match(blob.name)
                if match:
                    # Get the path without trailing slash
                    video_path = match.group(0).rstrip('/')
                    videos.add(video_path)
            
            video_list = sorted(list(videos), key=lambda x: int(x.split('_')[-1]))
            self.logger.info(f"Found videos: {video_list}")
            return video_list
            
        except Exception as e:
            self.logger.error(f"Error listing videos in {playlist_prefix}: {str(e)}")
            raise

    def _determine_role(self, speaker: str) -> str:
        """
        Determine the role (therapist or client) based on speaker identifier.
        
        Args:
            speaker (str): The speaker identifier from the transcript
            
        Returns:
            str: 'therapist' if speaker is 'Speaker A', 'client' otherwise
        """
        return "therapist" if speaker == "Speaker A" else "client"

    def _update_conversation_statistics(self, role: str, text: str, video_id: str):
        """
        Update conversation statistics based on speaker role.
        
        Args:
            role (str): The role of the speaker ('therapist' or 'client')
            text (str): The transcript text to analyze
            video_id (str): Identifier for the current video
            
        Note:
            Updates class statistics for word counts and conversation turns
        """
        word_count = self.count_words(text)
        if role == "therapist":
            self.statistics.num_therapist_convs += 1
            self.statistics.therapist_words.append(word_count)
            self.statistics.therapist_turns_per_video[video_id] += 1
        else:
            self.statistics.num_client_convs += 1
            self.statistics.client_words.append(word_count)
            self.statistics.client_turns_per_video[video_id] += 1

    def _update_duration_statistics(self, duration: int, is_corrupted: bool = False) -> str:
        """
        Update duration-related statistics and classify clip length.
        
        Args:
            duration (int): Duration of the clip in milliseconds
            is_corrupted (bool): Whether the clip is corrupted
            
        Returns:
            str: 'Yes' if clip is short (< 3 seconds), 'No' otherwise
        """
        self.statistics.total_duration_ms += duration
        is_short = "Yes" if duration/1000 < 3 else "No"
        
        if is_corrupted:
            self.statistics.corrupted_clips_duration_ms += duration
        elif is_short == "Yes":
            self.statistics.num_short_clips += 1
            self.statistics.short_clips_duration_ms += duration
        else:
            self.statistics.num_long_clips += 1
            self.statistics.long_clips_duration_ms += duration
            self.statistics.num_normal_clips += 1
            self.statistics.normal_clips_duration_ms += duration
                
        return is_short

    def _generate_clip_path(self, gcs_transcript_path: str, iterator: str) -> str:
        """
        Generate the Google Cloud Storage path for a clip.
        
        Args:
            gcs_transcript_path (str): Path to the transcript JSON file
            iterator (str): Formatted string representing clip number (e.g., '001')
            
        Returns:
            str: Complete GCS path to the clip file
            
        Note:
            Returns path in format: 'gs://bucket-name/path/to/clips/clip_XXX.mp4'
        """
        clips_dir = os.path.dirname(gcs_transcript_path)
        clip_path = f"{self.bucket_name}/{clips_dir}/clips/clip_{iterator}.mp4"
        return "gs://" + clip_path.replace('//', '/')

    def _handle_corruption_check(self, clip_path: str) -> Tuple[str, str]:
        """
        Check if a clip is corrupted and update corruption statistics.
        
        Args:
            clip_path (str): GCS path to the clip file
            
        Returns:
            Tuple[str, str]: 
                - First element: 'Yes' if corrupted, 'No' if not
                - Second element: Error message describing the corruption (empty if not corrupted)
            
        Note:
            Updates class statistics for corrupted clips and corruption reasons
        """
        is_corrupted, error_message = self.check_video_corruption(clip_path)
        
        if is_corrupted:
            self.statistics.num_corrupted_clips += 1
            self.statistics.corruption_reasons[error_message] += 1
            self.logger.warning(f"Corrupted clip {clip_path}: {error_message}")
            
        return ("Yes" if is_corrupted else "No", error_message)

    def _create_processed_entry(self, entry: Dict[str, Any], role: str, clip_path: str, 
                              is_short: str, corrupted: str, corruption_reason: str, 
                              transcript_entries: List[Dict[str, Any]], idx: int) -> Dict[str, Any]:
        """
        Create a processed entry dictionary containing all clip information.
        
        Args:
            entry (Dict[str, Any]): Original transcript entry
            role (str): Speaker role ('therapist' or 'client')
            clip_path (str): GCS path to the clip file
            is_short (str): 'Yes' if clip is short, 'No' otherwise
            corrupted (str): 'Yes' if clip is corrupted, 'No' otherwise
            corruption_reason (str): Description of corruption if present
            transcript_entries (List[Dict[str, Any]]): List of all transcript entries
            idx (int): Index of current entry in transcript
            
        Returns:
            Dict[str, Any]: Processed entry containing all relevant clip information
            including path, role, transcripts, analysis flags, and corruption status
        """
        return {
            "path": clip_path,
            "role": role,
            "short_clip": is_short,
            "corrupted": corrupted,
            "corruption_reason": corruption_reason if corrupted == "Yes" else "",
            "previous_transcript": self.build_previous_transcript(transcript_entries, idx),
            "transcript": entry["text"],
            "emotion": "",
            "strategy": "" if role == "therapist" else "NA",
            "analysis": "NA" if (is_short == "Yes" or role == "therapist" or corrupted == "Yes") else ""
        }

    def process_transcript_file(self, gcs_transcript_path: str, video_id: str) -> List[Dict[str, Any]]:
        """Process a single transcript file from GCS"""
        self.logger.info(f"Processing transcript: {gcs_transcript_path}")
        transcript_entries = self.read_json_from_gcs(gcs_transcript_path)
        processed_data = []
        
        for idx, entry in enumerate(transcript_entries):
            # Process entry
            role = self._determine_role(entry["speaker"])
            iterator = f"{idx + 1:03d}"
            
            # Process clip
            clip_path = self._generate_clip_path(gcs_transcript_path, iterator)
            corrupted, corruption_reason = self._handle_corruption_check(clip_path)
            
            # Update statistics with corruption information
            is_short = self._update_duration_statistics(entry["end"] - entry["start"], corrupted == "Yes")
            self._update_conversation_statistics(role, entry["text"], video_id)
            
            # Create and store processed entry
            processed_entry = self._create_processed_entry(
                entry, role, clip_path, is_short, corrupted, 
                corruption_reason, transcript_entries, idx
            )
            processed_data.append(processed_entry)
            
        self.logger.info(f"Processed {len(processed_data)} entries from transcript")
        return processed_data

    def build_previous_transcript(self, 
                                transcript_entries: List[Dict[Any, Any]], 
                                current_index: int) -> str:
        """
        Build previous transcript string from transcript entries.
        
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
        Process all transcripts in all playlists and videos.
        
        Returns:
            List[Dict[str, Any]]: List of all processed transcript entries
            
        Raises:
            google.cloud.exceptions.NotFound: If bucket doesn't exist
            google.api_core.exceptions.GoogleAPIError: For GCS API errors
            
        Note:
            This method processes all transcripts in the bucket, updating statistics
            and returning a list of processed entries. It handles errors for individual
            transcripts while continuing to process others.
        """
        all_processed_data = []
        
        # Get all playlists
        self.logger.info("Starting to process all transcripts")
        playlist_prefixes = self.list_playlists()
        if not playlist_prefixes:
            self.logger.warning("No playlists found in bucket!")
            return all_processed_data
            
        self.statistics.num_playlists = len(playlist_prefixes)
        self.logger.info(f"Found {self.statistics.num_playlists} playlists")
        
        for playlist_prefix in sorted(playlist_prefixes):
            self.logger.info(f"Processing playlist: {playlist_prefix}")
            # Get all videos in playlist
            video_prefixes = self.list_videos(playlist_prefix)
            if not video_prefixes:
                self.logger.warning(f"No videos found in playlist {playlist_prefix}")
                continue
                
            self.statistics.num_videos += len(video_prefixes)
            self.logger.info(f"Found {len(video_prefixes)} videos in {playlist_prefix}")
            
            for video_prefix in sorted(video_prefixes):
                self.logger.info(f"Processing video: {video_prefix}")
                transcript_path = f"{video_prefix}/transcript_timestamps.json"
                
                try:
                    video_data = self.process_transcript_file(transcript_path, video_prefix)
                    self.logger.info(f"Successfully processed {len(video_data)} clips from {transcript_path}")
                    all_processed_data.extend(video_data)
                except Exception as e:
                    self.logger.error(f"Error processing transcript at {transcript_path}: {str(e)}")
                    continue
        
        self.statistics.num_total_clips = len(all_processed_data)
        self.logger.info(f"Total processed clips: {self.statistics.num_total_clips}")
        return all_processed_data

    def save_to_json(self, output_path: str):
        """
        Process all transcripts and save results to JSON file.
        
        Args:
            output_path (str): Path where the output JSON file should be saved
            
        Raises:
            IOError: If there's an error creating the output directory or writing the file
            google.cloud.exceptions.NotFound: If bucket doesn't exist
            google.api_core.exceptions.GoogleAPIError: For GCS API errors
            
        Note:
            This method processes all transcripts, generates statistics, and saves
            the results to a JSON file. If the output directory doesn't exist,
            it will be created. The method also ensures proper JSON formatting
            and UTF-8 encoding.
        """
        if not output_path.endswith('.json'):
            output_path = f"{output_path}.json"
            
        self.logger.info("Starting transcript processing")
        processed_data = self.process_all_transcripts()
        
        if not processed_data:
            self.logger.warning("No data was processed. Nothing to save.")
            return
            
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self.logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                self.logger.critical(f"Error creating output directory: {str(e)}")
                raise
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Successfully processed transcripts and saved to {output_path}")
            
            # Print statistics
            self.statistics.print_statistics()
            
        except Exception as e:
            self.logger.critical(f"Error saving JSON file: {str(e)}")
            raise