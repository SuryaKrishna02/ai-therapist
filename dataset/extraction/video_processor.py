import os
import json
import yt_dlp
import subprocess
import assemblyai as aai
from typing import List, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class YouTubeAPI:
    """
    Handles interactions with the YouTube Data API.
    
    Attributes:
        api_key (str): YouTube Data API key for authentication
        youtube: YouTube API service object
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the YouTube API client.
        
        Args:
            api_key (str): YouTube Data API key
        """
        self.api_key = api_key
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def get_video_ids_from_playlist(self, playlist_id: str, playlist_dir: str) -> List[str]:
        """
        Retrieve all video IDs from a YouTube playlist with caching support.
        
        Args:
            playlist_id (str): YouTube playlist ID
            playlist_dir (str): Directory to cache video IDs
            
        Returns:
            List[str]: List of video IDs from the playlist
        """
        video_ids_file = os.path.join(playlist_dir, 'video_ids.json')
        
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

class VideoProcessor:
    """
    Handles video downloading, conversion, and clip extraction operations.
    
    Attributes:
        target_fps (int): Target frame rate for video conversion
    """
    
    def __init__(self, target_fps: int = 16):
        """
        Initialize the VideoProcessor.
        
        Args:
            target_fps (int, optional): Target frame rate for video conversion. Defaults to 16.
        """
        self.target_fps = target_fps

    @staticmethod
    def get_video_fps(video_file: str) -> Optional[float]:
        """
        Get the frame rate of a video file.
        
        Args:
            video_file (str): Path to video file
            
        Returns:
            Optional[float]: Frame rate of the video, or None if unable to determine
        """
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=r_frame_rate',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            fps = result.stdout.strip()
            if fps:
                num, denom = map(int, fps.split('/'))
                return num / denom
            return None
        except Exception as e:
            print(f"FPS detection failed for {video_file}: {e}")
            return None

    def download_and_convert_video(self, video_id: str, output_dir: str) -> bool:
        """
        Download and convert a YouTube video to the target frame rate.
        
        Args:
            video_id (str): YouTube video ID
            output_dir (str): Directory to save the processed video
            
        Returns:
            bool: True if successful, False otherwise
        """
        video_file = os.path.join(output_dir, f'{video_id}.mp4')
        temp_file = os.path.join(output_dir, f'temp_{video_id}.mp4')
        
        os.makedirs(output_dir, exist_ok=True)

        # Check existing video
        if os.path.exists(video_file):
            current_fps = self.get_video_fps(video_file)
            if current_fps and round(current_fps) == self.target_fps:
                print(f"Video already at target {self.target_fps}fps")
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
                print(f"Download failed for video {video_id}: {e}")
                return False

        # Convert frame rate
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', video_file,
                '-r', str(self.target_fps),
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'fast',
                '-c:a', 'aac',
                '-strict', 'experimental',
                temp_file
            ], check=True)
            
            os.replace(temp_file, video_file)
            
            # Verify conversion
            if self.get_video_fps(video_file) is None:
                print(f"Video {video_id} corrupted during conversion")
                return False
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Conversion failed for video {video_id}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

class AudioProcessor:
    """
    Handles audio extraction and transcription operations.
    
    Attributes:
        aai_api_key (str): AssemblyAI API key for transcription services
    """
    
    def __init__(self, aai_api_key: str):
        """
        Initialize the AudioProcessor.
        
        Args:
            aai_api_key (str): AssemblyAI API key
        """
        self.aai_api_key = aai_api_key
        aai.settings.api_key = aai_api_key

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """
        Extract audio from a video file.
        
        Args:
            video_path (str): Path to source video file
            audio_path (str): Path to save extracted audio
            
        Returns:
            bool: True if successful, False otherwise
        """
        if os.path.exists(audio_path):
            print(f"Audio already extracted for {os.path.basename(video_path)}")
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
            print(f"Audio extraction failed: {e}")
            return False

    def transcribe_audio(self, audio_file: str, txt_output: str, json_output: str) -> bool:
        """
        Transcribe audio with speaker diarization using AssemblyAI.
        
        Args:
            audio_file (str): Path to audio file
            txt_output (str): Path to save text transcript
            json_output (str): Path to save JSON transcript with timestamps
            
        Returns:
            bool: True if successful, False otherwise
        """
        if os.path.exists(json_output) and os.path.exists(txt_output):
            print(f"Transcripts already exist for {audio_file}")
            return True

        try:
            config = aai.TranscriptionConfig(
                speaker_labels=True,
                speakers_expected=2,
                punctuate=True,
                format_text=True
            )
            
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_file, config=config)

            # Save text transcript
            with open(txt_output, 'w', encoding='utf-8') as f:
                for utterance in transcript.utterances:
                    f.write(f"Speaker {utterance.speaker}: {utterance.text}\n")

            # Save JSON transcript
            diarized_output = [{
                'speaker': f"Speaker {u.speaker}",
                'start': u.start,
                'end': u.end,
                'text': u.text
            } for u in transcript.utterances]
            
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(diarized_output, f, ensure_ascii=False, indent=4)

            return True
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            return False

class VideoClipExtractor:
    """
    Handles extraction of video clips based on transcription timestamps.
    """
    
    @staticmethod
    def get_video_duration(video_file: str) -> Optional[float]:
        """
        Get the duration of a video file.
        
        Args:
            video_file (str): Path to video file
            
        Returns:
            Optional[float]: Duration in seconds, or None if unable to determine
        """
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Duration detection failed for {video_file}: {e}")
            return None

    def extract_clips(self, video_path: str, diarization_json: str, 
                     output_dir: str, min_duration: float = 3.0) -> bool:
        """
        Extract video clips based on transcription timestamps.
        
        Args:
            video_path (str): Path to source video
            diarization_json (str): Path to diarization JSON file
            output_dir (str): Directory to save extracted clips
            min_duration (float, optional): Minimum clip duration in seconds. Defaults to 3.0.
            
        Returns:
            bool: True if at least one clip was extracted successfully
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with open(diarization_json, 'r', encoding='utf-8') as f:
                segments = json.load(f)
        except Exception as e:
            print(f"Failed to load diarization data: {e}")
            return False

        success = False
        for idx, segment in enumerate(segments, 1):
            clip_path = os.path.join(output_dir, f"clip_{idx:03d}.mp4")
            
            # Skip if valid clip exists
            if os.path.exists(clip_path):
                duration = self.get_video_duration(clip_path)
                if duration and duration >= min_duration:
                    continue

            # Calculate timing
            start_sec = segment['start'] / 1000
            duration_sec = (segment['end'] - segment['start']) / 1000
            
            if duration_sec < min_duration:
                continue

            try:
                subprocess.run([
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-ss', str(start_sec),
                    '-t', str(duration_sec),
                    '-c', 'copy',
                    clip_path
                ], check=True)
                success = True
            except subprocess.CalledProcessError as e:
                print(f"Failed to extract clip {idx}: {e}")
                continue

        return success

class YouTubeProcessor:
    """
    Main class that orchestrates the entire YouTube video processing pipeline.
    
    Attributes:
        youtube_api (YouTubeAPI): YouTube API client
        video_processor (VideoProcessor): Video processing handler
        audio_processor (AudioProcessor): Audio processing handler
        clip_extractor (VideoClipExtractor): Clip extraction handler
        data_dir (str): Base directory for all processed data
    """
    
    def __init__(self, youtube_api_key: str, assemblyai_api_key: str, data_dir: str = 'data'):
        """
        Initialize the YouTube processing pipeline.
        
        Args:
            youtube_api_key (str): YouTube Data API key
            assemblyai_api_key (str): AssemblyAI API key
            data_dir (str, optional): Base directory for processed data. Defaults to 'data'.
        """
        self.youtube_api = YouTubeAPI(youtube_api_key)
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor(assemblyai_api_key)
        self.clip_extractor = VideoClipExtractor()
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)

    def process_playlist(self, playlist_id: str, playlist_idx: int) -> None:
        """
        Process all videos in a YouTube playlist.
        
        Args:
            playlist_id (str): YouTube playlist ID
            playlist_idx (int): Index for playlist directory naming
        """
        playlist_dir = os.path.join(self.data_dir, f'playlist_{playlist_idx}')
        os.makedirs(playlist_dir, exist_ok=True)
        
        video_ids = self.youtube_api.get_video_ids_from_playlist(playlist_id, playlist_dir)
        print(f"Processing {len(video_ids)} videos from playlist {playlist_id}")
        
        for vid_idx, video_id in enumerate(video_ids, 1):
            self.process_video(video_id, playlist_dir, vid_idx)

    def process_video(self, video_id: str, playlist_dir: str, video_idx: int) -> None:
        """
        Process a single video through the entire pipeline.
        
        Args:
            video_id (str): YouTube video ID
            playlist_dir (str): Playlist directory path
            video_idx (int): Index for video directory naming
        """
        video_dir = os.path.join(playlist_dir, f'video_{video_idx}')
        os.makedirs(video_dir, exist_ok=True)
        
        print(f"\nProcessing video {video_idx}: {video_id}")
        
        # Setup file paths
        video_file = os.path.join(video_dir, f'{video_id}.mp4')
        audio_file = os.path.join(video_dir, f'{video_id}.mp3')
        transcript_txt = os.path.join(video_dir, 'transcript.txt')
        transcript_json = os.path.join(video_dir, 'transcript_timestamps.json')
        clips_dir = os.path.join(video_dir, 'clips')
        
        # Step 1: Download and convert video
        if not self.video_processor.download_and_convert_video(video_id, video_dir):
            print(f"Failed to process video {video_id}. Skipping remaining steps.")
            return
            
        # Step 2: Extract audio
        if not self.audio_processor.extract_audio(video_file, audio_file):
            print(f"Failed to extract audio from {video_id}. Skipping remaining steps.")
            return
            
        # Step 3: Transcribe audio
        if not self.audio_processor.transcribe_audio(audio_file, transcript_txt, transcript_json):
            print(f"Failed to transcribe audio from {video_id}. Skipping remaining steps.")
            return
            
        # Step 4: Extract clips
        if not self.clip_extractor.extract_clips(video_file, transcript_json, clips_dir):
            print(f"Failed to extract clips from {video_id}")
            return
            
        print(f"Successfully processed video {video_id}")