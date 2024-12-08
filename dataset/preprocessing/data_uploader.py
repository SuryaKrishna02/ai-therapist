import os
import logging
import colorlog
from tqdm import tqdm
from google.cloud import storage

class GCSUploader:
    """
    A utility class for uploading files to Google Cloud Storage with progress tracking
    and colored logging.

    Attributes:
        logger (logging.Logger): Configured logger for logging events.
    """

    def __init__(self):
        """
        Initializes the GCSUploader class and sets up the logger.
        """
        self.logger = self.setup_logger()

    def setup_logger(self) -> logging.Logger:
        """
        Configure and return a logger with colored output.

        Returns:
            logging.Logger: Configured logger instance.
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

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        # Remove duplicate handlers if any
        if len(logger.handlers) > 1:
            logger.handlers = [handler]

        return logger

    def upload_files(self, local_directory: str, bucket_name: str):
        """
        Upload all files from the specified local directory to the specified Google Cloud Storage bucket.

        Args:
            local_directory (str): Path to the local directory containing files to upload.
            bucket_name (str): Name of the Google Cloud Storage bucket.

        Logs:
            INFO: When successfully connected to the bucket or uploaded a file.
            WARNING: If no files are found in the directory.
            ERROR: If there are issues connecting to the bucket or uploading files.
        """
        try:
            # Initialize Google Cloud Storage client
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            self.logger.info(f"Connected to bucket: {bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to bucket: {bucket_name}. Error: {e}")
            return

        # Collect files and calculate total size
        files_to_upload = []
        total_size = 0
        for root, _, files in os.walk(local_directory):
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                files_to_upload.append(local_file_path)
                total_size += os.path.getsize(local_file_path)

        if not files_to_upload:
            self.logger.warning(f"No files found in directory: {local_directory}")
            return

        self.logger.info(f"Starting upload of {len(files_to_upload)} files ({total_size / 1e6:.2f} MB)")

        # Initialize progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Uploading") as progress_bar:
            for local_file_path in files_to_upload:
                try:
                    # Define blob name (preserving directory structure)
                    blob_name = os.path.relpath(local_file_path, local_directory).replace("\\", "/")
                    blob = bucket.blob(blob_name)

                    # Upload file
                    file_size = os.path.getsize(local_file_path)
                    blob.upload_from_filename(local_file_path)

                    # Update progress bar
                    progress_bar.update(file_size)
                    self.logger.info(f"Uploaded {local_file_path} to {bucket_name}/{blob_name}")
                except Exception as e:
                    self.logger.error(f"Failed to upload {local_file_path}. Error: {e}")

        self.logger.info("Upload process completed.")