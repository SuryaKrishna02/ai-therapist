import asyncio
from annotate import TranscriptAnnotator
from postprocessor import DatasetConverter
from preprocessor import TranscriptProcessor
from constants import (
    PROJECT_ID,
    DATA_BUCKET_NAME,
    REPO_NAME,
    HF_TOKEN,
    COMMIT_MESSAGE
)

processed_transcripts_path = "./tmp/processed_transcripts.json"
annotated_transcripts_path = "./tmp/annotated_transcripts.json"

try:
    pre_processor = TranscriptProcessor(
        bucket_name=DATA_BUCKET_NAME, 
        project_id=PROJECT_ID
        )
    pre_processor.save_to_json(processed_transcripts_path)
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

finally:
    # Clean up resources if needed
    if hasattr(annotator, '_cleanup_llama_model'):
        annotator._cleanup_llama_model()
    if hasattr(annotator, '_cleanup_videollama_model'):
        annotator._cleanup_videollama_model()

try:
    # Initialize processor with default parameters from constants
    post_processor = DatasetConverter()
    post_processor.process_and_push_to_hf(
        input_file=annotated_transcripts_path, 
        repo_name=REPO_NAME,
        hf_token=HF_TOKEN,
        commit_message=COMMIT_MESSAGE
        )
    
except Exception as e:
    print(f"Dataset converstion failed: {str(e)}")