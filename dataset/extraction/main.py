from data_uploader import GCSUploader
from video_processor import YouTubeProcessor
from constants import (
    DATA_DIR, 
    DATA_BUCKET_NAME,
    YOUTUBE_API_KEY,
    ASSEMBLYAI_API_KEY,
    PLAYLIST_IDS
)

try:
    processor = YouTubeProcessor(
        youtube_api_key=YOUTUBE_API_KEY, 
        assemblyai_api_key=ASSEMBLYAI_API_KEY,
        data_dir=DATA_DIR
        )
    playlists = PLAYLIST_IDS
    for idx, playlist_id in enumerate(playlists, 1):
        print(f"\nProcessing playlist {idx}: {playlist_id}")
        try:
            processor.process_playlist(playlist_id, idx)
        except Exception as e:
            print(f"Error processing playlist {playlist_id}: {e}")
            continue
    
    print("\nAll processing completed.")
except Exception as e:
    print(f"Error Processing playlist: {str(e)}")

try:
    uploader = GCSUploader()
    uploader.upload_files(
        local_dir=DATA_DIR, 
        gcs_bucket=DATA_BUCKET_NAME
        )
except Exception as e:
    print(f"Error Uploading files: {str(e)}")