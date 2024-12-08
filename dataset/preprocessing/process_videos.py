import os
import json
import yt_dlp
import subprocess
import time
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import assemblyai as aai


def get_video_ids_from_playlist(youtube, playlist_id, playlist_dir):
    video_ids_file = os.path.join(playlist_dir, 'video_ids.json')
    if os.path.exists(video_ids_file):
        print(
            f"Video IDs already retrieved for playlist {
                playlist_id}. Loading from file."
        )
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


def get_video_fps(video_file):
    """
    Retrieve the frame rate of a video using ffprobe.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        fps = result.stdout.strip()
        if fps:
            num, denom = map(int, fps.split('/'))
            return num / denom
        else:
            return None
    except Exception as e:
        print(f"Failed to retrieve FPS for {video_file}: {e}")
        return None


def download_and_convert_video(video_id, output_dir, target_fps=16):
    video_file = os.path.join(output_dir, f'{video_id}.mp4')
    temp_file = os.path.join(output_dir, f'temp_{video_id}.mp4')

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Check if the video exists
    if os.path.exists(video_file):
        print(f"Video {video_id} already exists. Checking FPS...")
        current_fps = get_video_fps(video_file)
        if current_fps is None:
            print(
                f"Could not determine FPS for {
                    video_file}. Possibly corrupted or no video track."
            )
            return False  # Indicates failure or corrupted
        if round(current_fps) == target_fps:
            print(
                f"Video {video_id} is already at {
                    target_fps}fps. No conversion needed."
            )
            return True
        print(
            f"Video {video_id} is at {
                current_fps}fps. Converting to {target_fps}fps..."
        )
    else:
        # Step 2: Download the video
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': video_file,
            'merge_output_format': 'mp4',
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        except yt_dlp.utils.DownloadError as e:
            print(f"Failed to download video {video_id}: {e}")
            return False

    # Step 3: Convert to target FPS
    try:
        subprocess.run([
            'ffmpeg',
            '-y',
            '-i', video_file,
            '-r', str(target_fps),  # Set the frame rate
            '-c:v', 'libx264',      # Use H.264 codec for video compression
            '-crf', '23',           # Set quality level
            '-preset', 'fast',      # Set encoding speed
            '-c:a', 'aac',          # Use AAC codec for audio
            '-strict', 'experimental',
            temp_file
        ], check=True)

        # Replace the original file with the converted one
        os.replace(temp_file, video_file)
        print(f"Video {video_id} successfully converted to {target_fps}fps.")

        # Double-check if video track exists after conversion
        final_fps = get_video_fps(video_file)
        if final_fps is None:
            print(
                f"Video {video_id} appears to have no video track. Marking as corrupted.")
            return False

        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert video {video_id} to {target_fps}fps: {e}")
        # Clean up temporary file in case of failure
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return False


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


def transcribe_audio_diarization(audio_file, output_txt_file, output_json_file):
    if os.path.exists(output_json_file) and os.path.exists(output_txt_file):
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
    # Enable word-level timestamps
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2,
        punctuate=True,
        format_text=True,
    )
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

    # Prepare data for JSON output
    diarized_output = []
    for utterance in transcript.utterances:
        entry = {
            'speaker': f"Speaker {utterance.speaker}",
            'start': utterance.start,  # Start time in milliseconds
            'end': utterance.end,      # End time in milliseconds
            'text': utterance.text
        }
        diarized_output.append(entry)

    # Save the output to a JSON file
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(diarized_output, f, ensure_ascii=False, indent=4)

    print(f"Diarized transcript with timestamps saved to {output_json_file}")


def get_video_duration(video_file):
    """
    Retrieve the duration of a video using ffprobe.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"Failed to retrieve duration for {video_file}: {e}")
        return None


def extract_video_clips(video_path, diarization_json_path, clips_output_dir):
    os.makedirs(clips_output_dir, exist_ok=True)

    # Load the diarization data
    with open(diarization_json_path, 'r', encoding='utf-8') as f:
        diarization_data = json.load(f)

    for idx, entry in enumerate(diarization_data, 1):
        start_ms = entry['start']
        end_ms = entry['end']
        speaker = entry['speaker']
        text = entry['text']

        # Convert milliseconds to seconds
        start_sec = start_ms / 1000.0
        duration_sec = (end_ms - start_ms) / 1000.0

        # Prepare the output filename
        clip_filename = os.path.join(clips_output_dir, f"clip_{idx:03d}.mp4")
        # Check if the file already exists
        if os.path.exists(clip_filename):
            existing_duration = get_video_duration(clip_filename)
            if existing_duration is not None and existing_duration >= 3:
                print(f"Skipping clip {idx}: File already exists with sufficient duration ({
                      existing_duration:.2f}s).")
                continue
            else:
                print(f"Deleting clip {
                      idx}: Existing file is too short ({existing_duration:.2f}s).")
                os.remove(clip_filename)
        # Skip short clips
        if (end_ms - start_ms) < 3000:
            print(f"Skipping clip {idx}: Duration less than 3 seconds.")
            continue
        command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-i', video_path,
            '-ss', str(start_sec),
            '-t', str(duration_sec),
            '-c', 'copy',  # Copy codecs (no re-encoding for speed)
            clip_filename
        ]
        try:
            subprocess.run(command, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Extracted clip {clip_filename} for {speaker}: {text}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to extract clip {idx}: {e.stderr.decode('utf-8')}")
            continue


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

    for playlist_idx, playlist_id in enumerate(playlist_ids, start=1):
        print(f"\nProcessing playlist {playlist_id}...")

        # Create directory for the playlist with the new naming scheme
        playlist_dir = os.path.join(data_dir, f'playlist_{playlist_idx}')
        os.makedirs(playlist_dir, exist_ok=True)

        # Retrieve video IDs
        video_ids = get_video_ids_from_playlist(
            youtube, playlist_id, playlist_dir)
        print(f"Retrieved {len(video_ids)} videos from playlist {playlist_id}")

        for vid_idx, video_id in enumerate(video_ids, 1):
            print(f"\nProcessing video {vid_idx}/{len(video_ids)}: {video_id}")

            # Create directory for the video with the new naming scheme
            video_dir = os.path.join(playlist_dir, f'video_{vid_idx}')
            os.makedirs(video_dir, exist_ok=True)

            # Paths to the files
            video_file = os.path.join(video_dir, f'{video_id}.mp4')
            audio_file = os.path.join(video_dir, f'{video_id}.mp3')
            transcript_file = os.path.join(
                video_dir, f'{video_id}_transcript.txt')
            transcript_file_timestamp = os.path.join(
                video_dir, f'{video_id}_transcript_timestamps.json')

            # Download video
            print(f"Downloading video for {video_id}...")
            try:
                success = download_and_convert_video(video_id, video_dir)
            except Exception as e:
                print(f"Failed to download video {video_id}: {e}")
                continue  # Skip to the next video

            # If download or conversion failed or video is corrupted, skip further steps
            if not success or not os.path.exists(video_file):
                print(
                    f"Video {video_id} is corrupted or failed to download. Skipping...")
                continue

            # Extract audio
            print(f"Extracting audio for {video_id}...")
            try:
                extract_audio(video_file, audio_file)
            except Exception as e:
                print(f"Failed to extract audio for {video_id}: {e}")
                continue

            # Transcribe audio with speaker diarization
            transcribe_audio_option = True
            if transcribe_audio_option:
                print(f"Transcribing audio for {video_id} using AssemblyAI...")
                try:
                    transcribe_audio_diarization(
                        audio_file, transcript_file, transcript_file_timestamp)
                except Exception as e:
                    print(f"Failed to transcribe audio for {video_id}: {e}")
                    continue

            # Extract clips from video based on transcription timestamps
            extract_clips_option = True
            if extract_clips_option:
                print(f"Extracting video clips for {
                      video_id} based on timestamps...")
                try:
                    clips_dir = os.path.join(video_dir, 'clips')
                    extract_video_clips(
                        video_file, transcript_file_timestamp, clips_dir)
                except Exception as e:
                    print(f"Failed to extract clips for {video_id}: {e}")
                    continue

    print("\nAll processes completed successfully.")


if __name__ == "__main__":
    main()
