import os
import json
import yt_dlp
import subprocess
import assemblyai as aai
from googleapiclient.discovery import build
from typing import List, Dict, Any, Optional
from googleapiclient.errors import HttpError

class YouTubeProcessor:
    """
    Main class for processing YouTube videos, including downloading, conversion,
    transcription, and clip extraction.
    
    Attributes:
        data_dir (str): Base directory for storing processed data
        youtube_api_key (str): API key for YouTube Data API
        assemblyai_api_key (str): API key for AssemblyAI transcription service
        youtube: YouTube API client instance
    """
    
    def __init__(self, data_dir: str = 'data', 
                 youtube_api_key: Optional[str] = None,
                 assemblyai_api_key: Optional[str] = None):
        """
        Initialize the VideoProcessor with necessary API keys and directories.
        
        Args:
            data_dir (str): Base directory for storing processed data
            youtube_api_key (str, optional): YouTube Data API key
            assemblyai_api_key (str, optional): AssemblyAI API key
        
        Raises:
            ValueError: If required API keys are not provided or found in environment
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Setup API keys
        self.youtube_api_key = youtube_api_key or os.environ.get('YOUTUBE_API_KEY')
        if not self.youtube_api_key:
            raise ValueError("YouTube API key must be provided or set in environment")
            
        self.assemblyai_api_key = assemblyai_api_key or os.environ.get('ASSEMBLYAI_API_KEY')
        if not self.assemblyai_api_key:
            raise ValueError("AssemblyAI API key must be provided or set in environment")
            
        # Initialize YouTube API client
        self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
        aai.settings.api_key = self.assemblyai_api_key

    def get_video_ids_from_playlist(self, playlist_id: str, playlist_dir: str) -> List[str]:
        """
        Retrieve all video IDs from a YouTube playlist, with caching.
        
        Args:
            playlist_id (str): YouTube playlist ID
            playlist_dir (str): Directory to cache video IDs
            
        Returns:
            List[str]: List of video IDs from the playlist
            
        Raises:
            HttpError: If there's an error accessing the YouTube API
        """
        video_ids_file = os.path.join(playlist_dir, 'video_ids.json')
        
        # Check cache first
        if os.path.exists(video_ids_file):
            print(f"Loading cached video IDs for playlist {playlist_id}")
            with open(video_ids_file, 'r') as f:
                return json.load(f)

        video_ids = []
        next_page_token = None

        while True:
            try:
                response = self.youtube.playlistItems().list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()
                
                video_ids.extend([
                    item['contentDetails']['videoId'] 
                    for item in response['items']
                ])
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except HttpError as e:
                print(f"YouTube API error: {e}")
                break

        # Cache results
        with open(video_ids_file, 'w') as f:
            json.dump(video_ids, f)

        return video_ids

    def get_video_metadata(self, video_file: str) -> Dict[str, Any]:
        """
        Retrieve video metadata using ffprobe.
        
        Args:
            video_file (str): Path to video file
            
        Returns:
            Dict containing:
                - fps (float): Frames per second
                - duration (float): Duration in seconds
                - has_video (bool): Whether video track exists
                - has_audio (bool): Whether audio track exists
        """
        try:
            # Get FPS
            fps_result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                 '-show_entries', 'stream=r_frame_rate', '-of', 
                 'default=noprint_wrappers=1:nokey=1', video_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            # Get duration
            duration_result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            # Check streams
            streams_result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', video_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            # Parse results
            fps = None
            if fps_result.stdout.strip():
                num, denom = map(int, fps_result.stdout.strip().split('/'))
                fps = num / denom
                
            duration = float(duration_result.stdout.strip()) if duration_result.stdout.strip() else None
            
            streams = json.loads(streams_result.stdout)['streams']
            has_video = any(s['codec_type'] == 'video' for s in streams)
            has_audio = any(s['codec_type'] == 'audio' for s in streams)
            
            return {
                'fps': fps,
                'duration': duration,
                'has_video': has_video,
                'has_audio': has_audio
            }
            
        except Exception as e:
            print(f"Error getting video metadata: {e}")
            return {
                'fps': None,
                'duration': None,
                'has_video': False,
                'has_audio': False
            }

    def download_and_convert_video(self, video_id: str, output_dir: str, 
                                 target_fps: int = 16) -> bool:
        """
        Download YouTube video and convert to target FPS.
        
        Args:
            video_id (str): YouTube video ID
            output_dir (str): Directory to save the video
            target_fps (int): Target frames per second
            
        Returns:
            bool: True if successful, False otherwise
        """
        video_file = os.path.join(output_dir, 'video.mp4')
        temp_file = os.path.join(output_dir, 'temp_video.mp4')
        os.makedirs(output_dir, exist_ok=True)
        
        # Check existing video
        if os.path.exists(video_file):
            metadata = self.get_video_metadata(video_file)
            if not metadata['has_video']:
                return False
            if round(metadata['fps']) == target_fps:
                return True
                
        # Download if needed
        if not os.path.exists(video_file):
            try:
                ydl_opts = {
                    'format': 'bestvideo+bestaudio/best',
                    'outtmpl': video_file,
                    'merge_output_format': 'mp4',
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            except Exception as e:
                print(f"Download error: {e}")
                return False
                
        # Convert FPS
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', video_file,
                '-r', str(target_fps),
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'fast',
                '-c:a', 'aac',
                '-strict', 'experimental',
                temp_file
            ], check=True)
            
            os.replace(temp_file, video_file)
            
            # Verify conversion
            metadata = self.get_video_metadata(video_file)
            return metadata['has_video']
            
        except subprocess.CalledProcessError as e:
            print(f"Conversion error: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """
        Extract audio from video file.
        
        Args:
            video_path (str): Path to video file
            audio_path (str): Path to save extracted audio
            
        Returns:
            bool: True if successful, False otherwise
        """
        if os.path.exists(audio_path):
            return True
            
        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'mp3',
                '-ar', '44100',
                '-ac', '2',
                audio_path
            ], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Audio extraction error: {e}")
            return False

    def transcribe_audio_diarization(self, audio_file: str, 
                                   output_txt_file: str,
                                   output_json_file: str) -> bool:
        """
        Transcribe audio with speaker diarization using AssemblyAI.
        
        Args:
            audio_file (str): Path to audio file
            output_txt_file (str): Path to save text transcript
            output_json_file (str): Path to save JSON transcript with timestamps
            
        Returns:
            bool: True if successful, False otherwise
        """
        if os.path.exists(output_json_file) and os.path.exists(output_txt_file):
            return True
            
        try:
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=2,
                punctuate=True,
                format_text=True,
            )
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_file, config=config)
            
            # Save text transcript
            with open(output_txt_file, 'w', encoding='utf-8') as f:
                for utterance in transcript.utterances:
                    speaker = f"Speaker {utterance.speaker}"
                    f.write(f"{speaker}: {utterance.text}\n")
                    
            # Save JSON transcript
            diarized_output = [
                {
                    'speaker': f"Speaker {u.speaker}",
                    'start': u.start,
                    'end': u.end,
                    'text': u.text
                }
                for u in transcript.utterances
            ]
            
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(diarized_output, f, ensure_ascii=False, indent=4)
                
            return True
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return False

    def extract_video_clips(self, video_path: str, diarization_json_path: str, 
                          clips_output_dir: str) -> List[str]:
        """
        Extract video clips based on transcript timestamps.
        
        Args:
            video_path (str): Path to video file
            diarization_json_path (str): Path to diarization JSON file
            clips_output_dir (str): Directory to save clips
            
        Returns:
            List[str]: List of paths to successfully extracted clips
        """
        os.makedirs(clips_output_dir, exist_ok=True)
        successful_clips = []
        
        with open(diarization_json_path, 'r', encoding='utf-8') as f:
            diarization_data = json.load(f)
            
        for idx, entry in enumerate(diarization_data, 1):
            start_ms = entry['start']
            end_ms = entry['end']
            duration_ms = end_ms - start_ms
            
            if duration_ms < 3000:  # Skip clips shorter than 3 seconds
                continue
                
            clip_filename = os.path.join(clips_output_dir, f"clip_{idx:03d}.mp4")
            
            # Check existing clip
            if os.path.exists(clip_filename):
                metadata = self.get_video_metadata(clip_filename)
                if metadata['duration'] and metadata['duration'] >= 3:
                    successful_clips.append(clip_filename)
                    continue
                os.remove(clip_filename)
                
            try:
                subprocess.run([
                    'ffmpeg', '-y',
                    '-ss', str(start_ms / 1000),
                    '-i', video_path,
                    '-t', str(duration_ms / 1000),
                    '-c:v', 'libx264',
                    '-crf', '23',
                    '-preset', 'fast',
                    '-c:a', 'aac',
                    clip_filename
                ], check=True)
                
                successful_clips.append(clip_filename)
                
            except subprocess.CalledProcessError as e:
                print(f"Error extracting clip {idx}: {e}")
                continue
                
        return successful_clips

    def process_playlist(self, playlist_id: str, playlist_idx: int) -> Dict[str, Any]:
        """
        Process an entire YouTube playlist.
        
        Args:
            playlist_id (str): YouTube playlist ID
            playlist_idx (int): Index number for the playlist
            
        Returns:
            Dict containing:
                - success (bool): Overall success status
                - videos_processed (int): Number of videos processed
                - videos_failed (int): Number of videos that failed
                - clips_extracted (int): Total number of clips extracted
        """
        stats = {
            'success': False,
            'videos_processed': 0,
            'videos_failed': 0,
            'clips_extracted': 0
        }
        
        playlist_dir = os.path.join(self.data_dir, f'playlist_{playlist_idx}')
        os.makedirs(playlist_dir, exist_ok=True)
        
        try:
            video_ids = self.get_video_ids_from_playlist(playlist_id, playlist_dir)
            
            for vid_idx, video_id in enumerate(video_ids, 1):
                video_dir = os.path.join(playlist_dir, f'video_{vid_idx}')
                os.makedirs(video_dir, exist_ok=True)
                
                # Set up file paths
                video_file = os.path.join(video_dir, 'video.mp4')
                audio_file = os.path.join(video_dir, 'audio.mp3')
                transcript_file = os.path.join(video_dir, 'transcript.txt')
                transcript_json = os.path.join(video_dir, 'transcript_timestamps.json')
                clips_dir = os.path.join(video_dir, 'clips')
                
                print(f"\nProcessing video {vid_idx}/{len(video_ids)}: {video_id}")
                
                try:
                    # Download and convert video
                    if not self.download_and_convert_video(video_id, video_dir):
                        print(f"Failed to process video {video_id}")
                        stats['videos_failed'] += 1
                        continue
                        
                    # Extract audio
                    if not self.extract_audio(video_file, audio_file):
                        print(f"Failed to extract audio for {video_id}")
                        stats['videos_failed'] += 1
                        continue
                        
                    # Transcribe audio
                    if not self.transcribe_audio_diarization(
                        audio_file, transcript_file, transcript_json):
                        print(f"Failed to transcribe audio for {video_id}")
                        stats['videos_failed'] += 1
                        continue
                        
                    # Extract clips
                    clips = self.extract_video_clips(
                        video_file, transcript_json, clips_dir)
                    stats['clips_extracted'] += len(clips)
                    stats['videos_processed'] += 1
                    
                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")
                    stats['videos_failed'] += 1
                    continue
                    
            stats['success'] = True
            return stats
            
        except Exception as e:
            print(f"Error processing playlist {playlist_id}: {e}")
            return stats

    def process_multiple_playlists(self, playlist_ids: List[str]) -> Dict[str, Any]:
        """
        Process multiple YouTube playlists.
        
        Args:
            playlist_ids (List[str]): List of YouTube playlist IDs
            
        Returns:
            Dict containing aggregated statistics:
                - total_playlists (int): Number of playlists processed
                - total_videos_processed (int): Total videos successfully processed
                - total_videos_failed (int): Total videos that failed processing
                - total_clips_extracted (int): Total clips extracted
                - failed_playlists (List[str]): List of failed playlist IDs
        """
        aggregate_stats = {
            'total_playlists': len(playlist_ids),
            'total_videos_processed': 0,
            'total_videos_failed': 0,
            'total_clips_extracted': 0,
            'failed_playlists': []
        }
        
        for playlist_idx, playlist_id in enumerate(playlist_ids, 1):
            print(f"\nProcessing playlist {playlist_idx}/{len(playlist_ids)}: {playlist_id}")
            
            try:
                stats = self.process_playlist(playlist_id, playlist_idx)
                
                if stats['success']:
                    aggregate_stats['total_videos_processed'] += stats['videos_processed']
                    aggregate_stats['total_videos_failed'] += stats['videos_failed']
                    aggregate_stats['total_clips_extracted'] += stats['clips_extracted']
                else:
                    aggregate_stats['failed_playlists'].append(playlist_id)
                    
            except Exception as e:
                print(f"Failed to process playlist {playlist_id}: {e}")
                aggregate_stats['failed_playlists'].append(playlist_id)
                
        return aggregate_stats