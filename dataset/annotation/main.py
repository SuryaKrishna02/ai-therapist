from preprocessor import TranscriptProcessor

base_path = "../data"

output_path = "processed_transcripts.json"

try:
    processor = TranscriptProcessor(base_path)
    processor.save_to_json(output_path)
except Exception as e:
    print(f"Error processing transcripts: {str(e)}")