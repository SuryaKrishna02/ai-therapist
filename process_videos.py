import os
import json
import yt_dlp
import cv2
import subprocess
import time
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import assemblyai as aai


def get_video_ids_from_playlist(youtube, playlist_id, playlist_dir):
    video_ids_file = os.path.join(playlist_dir, 'video_ids.json')
    if os.path.exists(video_ids_file):
        print(f"Video IDs already retrieved for playlist {
              playlist_id}. Loading from file.")
        with open(video_ids_file, 'r') as f:
            video_ids = json.load(f)
        return video_ids

    video_ids = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        try:
            response = request.execute()
        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            break

        video_ids.extend([item['contentDetails']['videoId']
                         for item in response['items']])

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    # Save video IDs to a JSON file
    with open(video_ids_file, 'w') as f:
        json.dump(video_ids, f)

    return video_ids


def download_video(video_id, output_dir):
    video_file = os.path.join(output_dir, f'{video_id}.mp4')
    if os.path.exists(video_file):
        print(f"Video {video_id} already downloaded. Skipping download.")
        return

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'srt',
        'merge_output_format': 'mp4',
    }

    url = f'https://www.youtube.com/watch?v={video_id}'

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        print(f"Failed to download video {video_id}: {e}")
        raise


def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        print(f"Audio already extracted for {
              os.path.basename(video_path)}. Skipping audio extraction.")
        return

    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-acodec', 'mp3',
        '-ar', '44100',
        '-ac', '2',
        audio_path
    ]
    subprocess.run(command, check=True)


def transcribe_audio_diarization(audio_file, output_txt_file):
    if os.path.exists(output_txt_file):
        print(f"Diarized transcript already exists for {
              audio_file}. Skipping transcription.")
        return

    # Set your AssemblyAI API key
    ASSEMBLYAI_API_KEY = os.environ.get(
        'ASSEMBLYAI_API_KEY', '9af21f3f8465492486d796fc8825cbe5')
    if not ASSEMBLYAI_API_KEY:
        raise ValueError(
            "Please set the ASSEMBLYAI_API_KEY environment variable.")

    aai.settings.api_key = ASSEMBLYAI_API_KEY

    print(f"Transcribing {audio_file} with speaker diarization...")
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(
        audio_file,
        config=config
    )

    # Generate diarized transcript
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for utterance in transcript.utterances:
            speaker = f"Speaker {utterance.speaker}"
            text = utterance.text
            f.write(f"{speaker}: {text}\n")

    print(f"Diarized transcript saved to {output_txt_file}")


def extract_frames(video_path, output_dir, frame_rate=1):
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Frames already extracted for {
              os.path.basename(video_path)}. Skipping frame extraction.")
        return

    os.makedirs(output_dir, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # Default fps if unable to get from video
    interval = max(int(fps / frame_rate), 1)

    while success:
        if count % interval == 0:
            frame_id = int(count / interval)
            frame_filename = os.path.join(
                output_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(frame_filename, image)
        success, image = vidcap.read()
        count += 1


def main():
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = os.environ.get(
        'YOUTUBE_API_KEY', 'AIzaSyAE1ad6U3fy-Q4TyoIkzWHhWTX8FSwF9-Y')

    if not DEVELOPER_KEY:
        raise ValueError(
            "Please set the YOUTUBE_API_KEY environment variable.")

    youtube = build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    playlist_ids = [
        "PLEJO7-F_VqlPA0GmswOTvR1xM5TCeFCfM",
        "PLG1WzYaHQeh6JPLZcavyPLiz0ST4K406B",
        "PLdlFfrVsmlvCcIf77MOdU3fa6uQqmtFwR",
        "PLdlFfrVsmlvBmd0-FiSi6nRm1RNoFMt4x"
    ]

    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

    for playlist_id in playlist_ids:
        print(f"\nProcessing playlist {playlist_id}...")

        # Create directory for the playlist
        playlist_dir = os.path.join(data_dir, playlist_id)
        os.makedirs(playlist_dir, exist_ok=True)

        # Step 2: Retrieve video IDs
        video_ids = get_video_ids_from_playlist(
            youtube, playlist_id, playlist_dir)

        print(f"Retrieved {len(video_ids)} videos from playlist {playlist_id}")

        for idx, video_id in enumerate(video_ids, 1):
            print(f"\nProcessing video {idx}/{len(video_ids)}: {video_id}")

            # Create directory for the video
            video_dir = os.path.join(playlist_dir, video_id)
            os.makedirs(video_dir, exist_ok=True)

            # Paths to the files
            video_file = os.path.join(video_dir, f'{video_id}.mp4')
            # Changed to .mp3 for AssemblyAI
            audio_file = os.path.join(video_dir, f'{video_id}.wav')
            transcript_file = os.path.join(video_dir, f'{video_id}.txt')
            frames_dir = os.path.join(video_dir, 'frames')

            # Step 3: Download video
            print(f"Downloading video for {video_id}...")
            try:
                download_video(video_id, video_dir)
            except Exception as e:
                print(f"Failed to download video {video_id}: {e}")
                continue  # Skip to the next video

            # Check if video file exists
            if not os.path.exists(video_file):
                print(f"Video file not found for {video_id}")
                continue

            # Step 4: Extract audio
            print(f"Extracting audio for {video_id}...")
            try:
                extract_audio(video_file, audio_file)
            except Exception as e:
                print(f"Failed to extract audio for {video_id}: {e}")
                continue

            # Step 5: Transcribe audio using AssemblyAI with speaker diarization
            transcribe_audio_option = True  # Set to False if you don't want to transcribe audio
            if transcribe_audio_option:
                print(f"Transcribing audio for {video_id} using AssemblyAI...")
                try:
                    transcribe_audio_diarization(audio_file, transcript_file)
                except Exception as e:
                    print(f"Failed to transcribe audio for {video_id}: {e}")
                    continue

            # Step 6: Extract frames (optional)
            extract_frames_option = True  # Set to False if you don't want to extract frames
            if extract_frames_option:
                print(f"Extracting frames for {video_id}...")
                try:
                    # Adjust frame_rate as needed
                    extract_frames(video_file, frames_dir, frame_rate=1)
                except Exception as e:
                    print(f"Failed to extract frames for {video_id}: {e}")
                    continue

    print("\nAll processes completed successfully.")


if __name__ == "__main__":
    main()
