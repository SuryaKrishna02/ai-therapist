from preprocessor import TranscriptProcessor

location = "us-central1"
project_id = "x-casing-442000-s1"
bucket_name = "ai-therapist-data"  # GCS bucket name
output_path = "processed_transcripts.json"

try:
    processor = TranscriptProcessor(
        bucket_name=bucket_name, 
        project_id=project_id
        )
    processor.save_to_json(output_path)
except Exception as e:
    print(f"Error processing transcripts: {str(e)}")