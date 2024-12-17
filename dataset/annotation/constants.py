from vertexai.generative_models import (
    HarmCategory,
    HarmBlockThreshold,
    SafetySetting,
    GenerationConfig
)

# Safety settings configuration
SAFETY_CONFIG = [
    SafetySetting(category=cat, threshold=HarmBlockThreshold.OFF)
    for cat in [
        HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH, 
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY
    ]
]

# Generation configuration
GENERATION_CONFIG = GenerationConfig(
    temperature=0,
    top_p=1.0,
    top_k=32,
    candidate_count=1,
    max_output_tokens=8192,
)

MAX_RETRIES = 3
CALLS_PER_MINUTE = 60
LOCATION = "us-central1"
PROJECT_ID = "x-casing-442000-s1"
MODEL_NAME = "gemini-1.5-flash-002"
DATA_BUCKET_NAME = "ai-therapist-data"
REPO_NAME = "SuryaKrishna02/therapy-instruct"
HF_TOKEN = "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"
COMMIT_MESSAGE = "Added therapy instruct dataset"

# Data processing configurations
CONTEXT_WINDOW = 5
TRAIN_RATIO = 0.9
VAL_RATIO = 0
TEST_RATIO = 0.1
RANDOM_SEED = 42

# Template configurations
INSTRUCTION_TEMPLATE = """\
You are a compassionate and skilled therapist with tons of experience.
Using past conversations context, for each client input: 
(1) Identify the client's emotion; 
(2) Reflect on your emotion as the therapist; 
(3) Determine a therapeutic strategy
(4) Provide a compassionate and helpful response.
"""

# File path patterns
CLIP_PATH_PATTERN = r'(playlist_\d+/video_\d+)'
CLIP_NUMBER_PATTERN = r'clip_(\d+)'