import asyncio
from annotate import TranscriptAnnotator
from preprocessor import TranscriptProcessor
from constants import (
    PROJECT_ID,
    DATA_BUCKET_NAME
)

processed_transcripts_path = "processed_transcripts.json"
annotated_transcripts_path = "annotated_transcripts.json"

try:
    processor = TranscriptProcessor(
        bucket_name=DATA_BUCKET_NAME, 
        project_id=PROJECT_ID
        )
    processor.save_to_json(processed_transcripts_path)
except Exception as e:
    print(f"Error processing transcripts: {str(e)}")

try:
    annotator = TranscriptAnnotator(
        use_local=True
    )

    asyncio.run(annotator.process_transcripts(
        input_file=processed_transcripts_path, 
        output_file=annotated_transcripts_path
        )
    )

except Exception as e:
    print(f"Error annotating videos: {str(e)}")