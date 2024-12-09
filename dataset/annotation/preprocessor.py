import re
import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import timedelta
from google.cloud import storage
from logger import setup_logging
from collections import defaultdict
from typing import List, Dict, Any, Tuple

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
        """
        self.logger = setup_logging(self.__class__.__name__)
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
            "Normal Clips": self._format_percentage(self.num_normal_clips, self.num_total_clips),
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
        """
        word_stats = self.calculate_word_stats()
        turn_stats = self.calculate_turn_stats()
        total_convs = self.num_therapist_convs + self.num_client_convs
        
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
                "Average Client Turns per Video": turn_stats["avg_client_turns"]
            },
            "Word Count Analysis": {
                "Average Words per Therapist Turn": word_stats["avg_therapist_words"],
                "Average Words per Client Turn": word_stats["avg_client_words"],
                "Average Words per Turn (Overall)": word_stats["avg_total_words"]
            }
        }
        
        self.logger.info("\n=== Dataset Statistics ===")
        for category, category_stats in stats.items():
            self.logger.info(f"\n{category}:")
            for key, value in category_stats.items():
                self.logger.info(f"  {key}: {value}")

class TranscriptProcessor:
    """
    A class for processing transcript files from either Google Cloud Storage or local directory.
    
    This class handles the extraction, analysis, and processing of transcript files stored
    either in Google Cloud Storage or in a local directory. It manages temporary file storage,
    corruption checking, and statistical analysis of the processed transcripts.
    
    Attributes:
        storage_type (str): Type of storage ('gcs' or 'local')
        base_path (str): Base path for local storage or bucket name for GCS
        storage_client (storage.Client): Google Cloud Storage client (if using GCS)
        bucket (storage.Bucket): GCS bucket instance (if using GCS)
        statistics (TranscriptStatistics): Statistics tracking instance
        temp_dir (Path): Path to temporary file storage directory
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(self, base_path: str, storage_type: str = 'local', project_id: str = None):
        """
        Initialize the TranscriptProcessor with either GCS bucket or local directory.
        
        Args:
            base_path (str): Base path for local storage or bucket name for GCS
            storage_type (str): Type of storage ('gcs' or 'local'). Defaults to 'local'
            project_id (str, optional): Google Cloud Project ID. Required for GCS.
        
        Raises:
            ValueError: If storage_type is invalid or requirements not met
            google.cloud.exceptions.NotFound: If GCS bucket doesn't exist
            google.auth.exceptions.DefaultCredentialsError: If GCP credentials are invalid
        """
        if storage_type not in ['gcs', 'local']:
            raise ValueError("storage_type must be either 'gcs' or 'local'")
            
        if storage_type == 'gcs' and project_id is None:
            raise ValueError("project_id is required for GCS storage")
            
        self.logger = setup_logging(self.__class__.__name__)
        self.storage_type = storage_type
        self.base_path = base_path
        self.file_not_found_error = "File Not Found"
        
        # Set up temporary directory
        self.temp_dir = Path("tmp")
        self.temp_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created temporary directory at {self.temp_dir}")
        
        # Initialize storage client if using GCS
        if storage_type == 'gcs':
            self.storage_client = storage.Client(project=project_id)
            self.bucket = self.storage_client.bucket(base_path)
            self.logger.info(f"Connected to GCS bucket: {base_path}")
        else:
            self.storage_client = None
            self.bucket = None
            if not os.path.exists(base_path):
                os.makedirs(base_path)
                self.logger.info(f"Created local directory: {base_path}")
            self.logger.info(f"Using local directory: {base_path}")
            
        self.statistics = TranscriptStatistics()

    def __del__(self):
        """
        Cleanup temporary files when the object is destroyed.
        
        This method ensures proper cleanup of temporary files and directories
        when the TranscriptProcessor instance is being destroyed.
        """
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {str(e)}")

    def read_json_from_storage(self, path: str) -> Dict:
        """
        Read JSON file from either GCS or local storage.
        
        Args:
            path (str): Path to the JSON file
            
        Returns:
            Dict: Parsed JSON content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            JSONDecodeError: If content is not valid JSON
        """
        try:
            self.logger.info(f"Reading JSON from: {path}")
            
            if self.storage_type == 'gcs':
                blob = self.bucket.blob(path)
                if not blob.exists():
                    raise FileNotFoundError(f"Blob does not exist: {path}")
                content = blob.download_as_text()
            else:
                full_path = os.path.join(self.base_path, path)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"File does not exist: {full_path}")
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
            return json.loads(content)
            
        except Exception as e:
            self.logger.error(f"Error reading JSON from {path}: {str(e)}")
            raise

    def get_file_path(self, relative_path: str) -> str:
        """
        Generate complete file path based on storage type.
        
        Args:
            relative_path (str): Relative path to the file
            
        Returns:
            str: Complete file path with appropriate prefix
        """
        if self.storage_type == 'gcs':
            return f"gs://{self.base_path}/{relative_path}"
        return os.path.join(self.base_path, relative_path)

    def count_words(self, text: str) -> int:
        """
        Count words in a text string.
        
        Args:
            text (str): The text to count words in
            
        Returns:
            int: Number of words in the text
        """
        return len(re.findall(r'\w+', text))

    def list_playlists(self) -> List[str]:
        """
        List all playlist folders in storage.
        
        Returns:
            List[str]: List of unique playlist prefixes
            
        Raises:
            google.cloud.exceptions.NotFound: If bucket doesn't exist (GCS)
            OSError: If base directory is not accessible (local)
        """
        try:
            self.logger.info("Listing playlists...")
            
            if self.storage_type == 'gcs':
                all_blobs = list(self.bucket.list_blobs())
                playlists = {
                    blob.name.split('/')[0] 
                    for blob in all_blobs 
                    if blob.name.startswith('playlist_')
                }
            else:
                playlists = {
                    d for d in os.listdir(self.base_path)
                    if os.path.isdir(os.path.join(self.base_path, d))
                    and d.startswith('playlist_')
                }
            
            playlist_list = sorted(list(playlists))
            self.logger.info(f"Found playlists: {playlist_list}")
            return playlist_list
            
        except Exception as e:
            self.logger.error(f"Error listing playlists: {str(e)}")
            raise

    def list_videos(self, playlist_prefix: str) -> List[str]:
        """
        List all video folders in a playlist.
        
        Args:
            playlist_prefix (str): The playlist prefix to search within
            
        Returns:
            List[str]: List of video folder paths matching the pattern
            
        Raises:
            FileNotFoundError: If playlist directory doesn't exist
        """
        try:
            self.logger.info(f"Listing videos in playlist: {playlist_prefix}")
            video_pattern = re.compile(r'video_\d+')
            
            if self.storage_type == 'gcs':
                blobs = list(self.bucket.list_blobs(prefix=f"{playlist_prefix}/"))
                videos = set()
                for blob in blobs:
                    parts = blob.name.split('/')
                    if len(parts) > 1 and video_pattern.match(parts[1]):
                        videos.add(f"{playlist_prefix}/{parts[1]}")
            else:
                playlist_path = os.path.join(self.base_path, playlist_prefix)
                if not os.path.exists(playlist_path):
                    raise FileNotFoundError(f"Playlist directory not found: {playlist_path}")
                    
                videos = {
                    f"{playlist_prefix}/{d}"
                    for d in os.listdir(playlist_path)
                    if os.path.isdir(os.path.join(playlist_path, d))
                    and video_pattern.match(d)
                }
            
            video_list = sorted(list(videos), key=lambda x: int(x.split('_')[-1]))
            self.logger.info(f"Found videos: {video_list}")
            return video_list
            
        except Exception as e:
            self.logger.error(f"Error listing videos in {playlist_prefix}: {str(e)}")
            raise

    def _get_clip_file_path(self, clip_path: str) -> Tuple[Path, bool]:
        """
        Get the file path for clip checking, handling both storage types.
        
        Args:
            clip_path (str): Original clip path
            
        Returns:
            Tuple[Path, bool]: (file path, is_temporary)
            
        Raises:
            FileNotFoundError: If clip file doesn't exist
        """
        if self.storage_type == 'gcs':
            clean_path = clip_path.replace(f"gs://{self.base_path}/", "")
            blob = self.bucket.blob(clean_path)
            if not blob.exists():
                raise FileNotFoundError(self.file_not_found_error)
            temp_file_path = self.temp_dir / f"temp_clip_{hash(clip_path)}.mp4"
            blob.download_to_filename(str(temp_file_path))
            return temp_file_path, True
        
        if not os.path.exists(clip_path):
            raise FileNotFoundError(self.file_not_found_error)
        return Path(clip_path), False

    def _validate_probe_data(self, probe_data: Dict) -> Tuple[bool, str]:
        """
        Validate video probe data for corruption.
        
        Args:
            probe_data (Dict): FFprobe output data
            
        Returns:
            Tuple[bool, str]: (is_corrupted, error_message)
        """
        streams = probe_data.get('streams', [])
        stream_info = self._analyze_streams(streams)
        
        if not stream_info['has_video']:
            return True, "No video stream found"
            
        if not stream_info['has_audio']:
            return True, "No audio stream found"
            
        if not self._check_video_dimensions(stream_info['video_info']):
            return True, "Invalid video dimensions"
            
        fps_validation = self._validate_framerate(stream_info['video_info'].get('fps', '0/1'))
        if fps_validation[0]:  # has error
            return True, fps_validation[1]
        
        return False, "Valid video file"

    def _analyze_streams(self, streams: List[Dict]) -> Dict:
        """
        Analyze video and audio streams from probe data.
        
        Args:
            streams (List[Dict]): List of stream data from FFprobe
            
        Returns:
            Dict: Stream analysis results
        """
        result = {
            'has_video': False,
            'has_audio': False,
            'video_info': {}
        }
        
        for stream in streams:
            codec_type = stream.get('codec_type')
            if codec_type == 'video':
                result['has_video'] = True
                result['video_info'] = {
                    'fps': stream.get('r_frame_rate', 'N/A'),
                    'width': stream.get('width', 0),
                    'height': stream.get('height', 0),
                    'codec': stream.get('codec_name', 'unknown')
                }
            elif codec_type == 'audio':
                result['has_audio'] = True
        
        return result

    def _check_video_dimensions(self, video_info: Dict) -> bool:
        """
        Check if video dimensions are valid.
        
        Args:
            video_info (Dict): Video stream information
            
        Returns:
            bool: True if dimensions are valid
        """
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        return width > 0 and height > 0

    def _validate_framerate(self, fps_str: str) -> Tuple[bool, str]:
        """
        Validate video framerate.
        
        Args:
            fps_str (str): Framerate string in format 'num/den'
            
        Returns:
            Tuple[bool, str]: (has_error, error_message)
        """
        try:
            num, den = map(int, fps_str.split('/'))
            fps = num / den if den != 0 else 0
            if fps <= 0:
                return True, f"Invalid framerate: {fps_str}"
            return False, ""
        except (ValueError, ZeroDivisionError):
            return True, f"Could not parse framerate: {fps_str}"

    def check_video_corruption(self, clip_path: str) -> Tuple[bool, str]:
        """
        Check if a video clip is corrupted using ffprobe.
        
        Args:
            clip_path (str): Path to the video clip
            
        Returns:
            Tuple[bool, str]: 
                - First element: True if corrupted, False if valid
                - Second element: Description of corruption if present
        """
        try:
            # Get file path and handle temporary file if needed
            temp_file_path, is_temporary = self._get_clip_file_path(clip_path)
            
            try:
                # Run ffprobe
                ffprobe_cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_streams',
                    '-show_format',
                    str(temp_file_path)
                ]
                
                result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return True, f"FFprobe error: {result.stderr}"
                
                probe_data = json.loads(result.stdout)
                return self._validate_probe_data(probe_data)
                
            finally:
                # Clean up temporary file if using GCS
                if is_temporary:
                    temp_file_path.unlink()
                    
        except FileNotFoundError:
            return False, self.file_not_found_error
        except json.JSONDecodeError:
            return True, "Failed to parse FFprobe output"
        except Exception as e:
            self.logger.error(f"Error checking video corruption for {clip_path}: {str(e)}")
            return True, f"Error checking video: {str(e)}"


    def _generate_clip_path(self, transcript_path: str, iterator: str) -> str:
        """
        Generate the storage path for a clip.
        
        Args:
            transcript_path (str): Path to the transcript JSON file
            iterator (str): Formatted string representing clip number
            
        Returns:
            str: Complete path to the clip file
        """
        clips_dir = os.path.dirname(transcript_path)
        clip_name = f"clips/clip_{iterator}.mp4"
        
        if self.storage_type == 'gcs':
            return f"gs://{self.base_path}/{clips_dir}/{clip_name}"
        return os.path.join(self.base_path, clips_dir, clip_name)

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

    def _create_processed_entry(self, entry: Dict[str, Any], role: str, clip_path: str, 
                              is_short: str, corrupted: str, corruption_reason: str, 
                              transcript_entries: List[Dict[str, Any]], idx: int) -> Dict[str, Any]:
        """
        Create a processed entry dictionary containing all clip information.
        
        Args:
            entry (Dict[str, Any]): Original transcript entry
            role (str): Speaker role ('therapist' or 'client')
            clip_path (str): Path to the clip file
            is_short (str): 'Yes' if clip is short, 'No' otherwise
            corrupted (str): 'Yes' if clip is corrupted, 'No' otherwise
            corruption_reason (str): Description of corruption if present
            transcript_entries (List[Dict[str, Any]]): List of all transcript entries
            idx (int): Index of current entry in transcript
            
        Returns:
            Dict[str, Any]: Processed entry containing all relevant clip information
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

    def build_previous_transcript(self, transcript_entries: List[Dict[str, Any]], 
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

    def process_transcript_file(self, gcs_transcript_path: str, video_id: str) -> List[Dict[str, Any]]:
        """
        Process a single transcript file.
        
        Args:
            gcs_transcript_path (str): Path to the transcript file
            video_id (str): Identifier for the video being processed
            
        Returns:
            List[Dict[str, Any]]: List of processed transcript entries
            
        Raises:
            FileNotFoundError: If transcript file doesn't exist
            JSONDecodeError: If transcript file is not valid JSON
        """
        self.logger.info(f"Processing transcript: {gcs_transcript_path}")
        transcript_entries = self.read_json_from_storage(gcs_transcript_path)
        processed_data = []
        
        for idx, entry in enumerate(transcript_entries):
            # Process entry and determine role
            role = self._determine_role(entry["speaker"])
            iterator = f"{idx + 1:03d}"
            
            # Generate clip path and check for corruption
            clip_path = self._generate_clip_path(gcs_transcript_path, iterator)
            corrupted, corruption_reason = self._handle_corruption_check(clip_path)
            
            # Update statistics
            is_short = self._update_duration_statistics(entry["end"] - entry["start"], 
                                                      corrupted == "Yes")
            self._update_conversation_statistics(role, entry["text"], video_id)
            
            # Create and store processed entry
            processed_entry = self._create_processed_entry(
                entry, role, clip_path, is_short, corrupted, 
                corruption_reason, transcript_entries, idx
            )
            processed_data.append(processed_entry)
            
        self.logger.info(f"Processed {len(processed_data)} entries from transcript")
        return processed_data

    def _handle_corruption_check(self, clip_path: str) -> Tuple[str, str]:
        """
        Check if a clip is corrupted and update corruption statistics.
        
        Args:
            clip_path (str): Path to the clip file
            
        Returns:
            Tuple[str, str]: 
                - First element: 'Yes' if corrupted, 'No' if not
                - Second element: Error message describing the corruption
        """
        is_corrupted, error_message = self.check_video_corruption(clip_path)
        
        if is_corrupted:
            self.statistics.num_corrupted_clips += 1
            self.statistics.corruption_reasons[error_message] += 1
            self.logger.warning(f"Corrupted clip {clip_path}: {error_message}")
            
        return ("Yes" if is_corrupted else "No", error_message)

    def process_all_transcripts(self) -> List[Dict[str, Any]]:
        """
        Process all transcripts in all playlists and videos.
        
        This method processes all transcripts in the storage location, updating statistics
        and returning a list of processed entries.
        
        Returns:
            List[Dict[str, Any]]: List of all processed transcript entries
            
        Raises:
            Exception: If there are errors accessing storage or processing transcripts
        """
        all_processed_data = []
        
        # Get all playlists
        self.logger.info("Starting to process all transcripts")
        playlist_prefixes = self.list_playlists()
        if not playlist_prefixes:
            self.logger.warning("No playlists found!")
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