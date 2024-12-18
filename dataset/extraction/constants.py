import os
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).parent.parent

# Load the .env file from root directory
load_dotenv(root_dir / '.env')

DATA_BUCKET_NAME = os.getenv("DATA_BUCKET_NAME")
DATA_DIR = "../data"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
PLAYLIST_IDS = [
    "PLEJO7-F_VqlPA0GmswOTvR1xM5TCeFCfM",
    "PLG1WzYaHQeh6JPLZcavyPLiz0ST4K406B",
    "PLdlFfrVsmlvCcIf77MOdU3fa6uQqmtFwR",
    "PLdlFfrVsmlvBmd0-FiSi6nRm1RNoFMt4x"
]