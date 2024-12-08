from data_uploader import GCSUploader
from constants import LOCAL_DIR, DATA_BUCKET_NAME


try:
    uploader = GCSUploader()
    uploader.upload_files(
        local_dir=LOCAL_DIR, 
        gcs_bucket=DATA_BUCKET_NAME
        )
except Exception as e:
    print(f"Error Uploading files: {str(e)}")