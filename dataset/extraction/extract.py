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

    stats = processor.process_multiple_playlists(
        playlist_ids=PLAYLIST_IDS
    )
    
    print("\n=== Processing Complete ===")
    print(f"Total Playlists Processed: {stats['total_playlists']}")
    print(f"Total Videos Successfully Processed: {stats['total_videos_processed']}")
    print(f"Total Videos Failed: {stats['total_videos_failed']}")
    print(f"Total Clips Extracted: {stats['total_clips_extracted']}")
    
    if stats['failed_playlists']:
        print("\nFailed Playlists:")
        for playlist_id in stats['failed_playlists']:
            print(f"- {playlist_id}")
except Exception as e:
    print(f"Error Processing playlists: {str(e)}")

try:
    uploader = GCSUploader()
    uploader.upload_files(
        local_dir=DATA_DIR, 
        gcs_bucket=DATA_BUCKET_NAME
        )
except Exception as e:
    print(f"Error Uploading files: {str(e)}")