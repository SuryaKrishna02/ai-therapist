import asyncio
from annotate import TranscriptAnnotator
from preprocessor import TranscriptProcessor
from constants import (
    PROJECT_ID,
    DATA_BUCKET_NAME,
    MODEL_NAME
)

output_path = "processed_transcripts.json"
input_transcripts_file = "sample_transcripts.json"
output_transcripts_file = "annotated_transcripts.json"

try:
    processor = TranscriptProcessor(
        bucket_name=DATA_BUCKET_NAME, 
        project_id=PROJECT_ID
        )
    processor.save_to_json(output_path)
except Exception as e:
    print(f"Error processing transcripts: {str(e)}")

# try:
#     # Initialize annotator
#     annotator = TranscriptAnnotator(
#         model_name=MODEL_NAME
#     )

#     # Run processing
#     asyncio.run(annotator.process_transcripts(input_transcripts_file, output_transcripts_file))

# except Exception as e:
#     print(f"Error annotating videos: {str(e)}")