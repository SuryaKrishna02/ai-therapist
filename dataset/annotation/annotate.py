import json
import backoff
import logging
import asyncio
import vertexai
import colorlog
from typing import Dict, Optional
from google.api_core import exceptions
from vertexai import generative_models
from ratelimit import limits, sleep_and_retry
from vertexai.generative_models import GenerativeModel, Part
from constants import (
    SAFETY_CONFIG,
    GENERATION_CONFIG,
    LOCATION,
    PROJECT_ID,
    CALLS_PER_MINUTE,
    MAX_RETRIES
)
from prompts import (
    VIDEO_TEXT_EMOTION_DETECTION,
    TEXT_EMOTION_DETECTION,
    VIDEO_TEXT_EMOTION_ANALYSIS,
    VIDEO_TEXT_STRATEGY,
    TEXT_STRATEGY,
    VIDEO_TEXT_EMOTION_TEMPLATE,
    TEXT_EMOTION_TEMPLATE,
    VIDEO_TEXT_ANALYSIS_TEMPLATE,
    STRATEGY_TEMPLATE,
)

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

class TranscriptAnnotator:
    """Class for annotating therapy session transcripts using various models"""
    
    def __init__(self, model_name: str = "gemini-1.5-flash-002"):
        """
        Initialize the TranscriptAnnotator
        
        Args:
            model_name (str): Name of the generative model to use
            calls_per_minute (int): Rate limit for API calls
            max_retries (int): Maximum number of retries for failed calls
        """
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        self.model_name = model_name
        self.setup_logger()
        
    def setup_logger(self):
        """Configure colored logging"""
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

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        # Remove any existing handlers to avoid duplicate logs
        for existing_handler in self.logger.handlers[:-1]:
            self.logger.removeHandler(existing_handler)

    def extract_json_field(self, response_text: str, field: str) -> str:
        """
        Extract specific field from JSON response text
        
        Args:
            response_text (str): Response text containing JSON
            field (str): Field to extract from JSON
            
        Returns:
            str: Extracted field value
            
        Raises:
            json.JSONDecodeError: If JSON parsing fails
            KeyError: If field not found in JSON
        """
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                raise json.JSONDecodeError("No JSON found in response", response_text, 0)
                
            json_str = response_text[start_idx:end_idx]
            response_dict = json.loads(json_str)
            return response_dict[field]
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error extracting field '{field}' from response: {str(e)}")
            raise

    @sleep_and_retry
    @limits(calls=CALLS_PER_MINUTE, period=60)
    @backoff.on_exception(
        backoff.expo,
        (exceptions.ServiceUnavailable, exceptions.InternalServerError),
        max_tries=MAX_RETRIES
    )
    async def call_model(self, 
                        system_instruction: str,
                        prompt: str,
                        video_path: Optional[str] = None) -> str:
        """
        Call the generative model with rate limiting and retries
        
        Args:
            system_instruction (str): System instruction for the model
            prompt (str): Prompt template filled with data
            video_path (Optional[str]): Path to video file if needed
            
        Returns:
            str: Model response text
            
        Raises:
            ModelError: If model call fails after retries
        """
        try:
            model = GenerativeModel(
                self.model_name,
                system_instruction=system_instruction,
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_CONFIG
            )

            contents = []
            if video_path:
                video_file = Part.from_uri(
                    uri=video_path,
                    mime_type="video/mp4"
                )
                contents.append(video_file)
            contents.append(prompt)

            response = model.generate_content(contents)
            return response.text

        except exceptions.InvalidArgument as e:
            # Handle 400 Bad Request errors
            self.logger.error(f"Bad request for video {video_path}: {str(e)}")
            with open('corrupted_files.txt', 'a') as f:
                f.write(f"{video_path}\n")
            raise

        except (exceptions.ServiceUnavailable, exceptions.InternalServerError) as e:
            # Handle 5xx server errors that should be retried
            self.logger.warning(f"Server error, retrying: {str(e)}")
            raise

        except exceptions.GoogleAPICallError as e:
            # Handle other API errors
            self.logger.error(f"API error for video {video_path}: {str(e)}")
            raise

    async def process_entry(self, entry: Dict) -> Dict:
        """
        Process a single transcript entry based on its role and clip type
        
        Args:
            entry (Dict): Transcript entry to process
            
        Returns:
            Dict: Processed entry with model predictions
        """
        try:
            role = entry["role"]
            is_short = entry["short_clip"] == "Yes"
            video_path = entry["path"]
            dialogue = entry["transcript"]
            context = entry["previous_transcript"]

            # Client clips
            if role == "client":
                if is_short:
                    # Text emotion detection for short client clips
                    prompt = TEXT_EMOTION_TEMPLATE.format(role=role, dialogue=dialogue)
                    response = await self.call_model(TEXT_EMOTION_DETECTION, prompt)
                    entry["emotion"] = self.extract_json_field(response, "emotion")
                else:
                    # Video-text emotion detection and analysis for long client clips
                    prompt = VIDEO_TEXT_EMOTION_TEMPLATE.format(dialogue=dialogue)
                    response = await self.call_model(VIDEO_TEXT_EMOTION_DETECTION, prompt, video_path)
                    entry["emotion"] = self.extract_json_field(response, "emotion")

                    response = await self.call_model(VIDEO_TEXT_EMOTION_ANALYSIS, VIDEO_TEXT_ANALYSIS_TEMPLATE, video_path)
                    entry["analysis"] = self.extract_json_field(response, "emotional_cues")

            # Therapist clips
            else:
                if is_short:
                    # Text emotion detection and strategy prediction for short therapist clips
                    prompt = TEXT_EMOTION_TEMPLATE.format(role=role, dialogue=dialogue)
                    response = await self.call_model(TEXT_EMOTION_DETECTION, prompt)
                    entry["emotion"] = self.extract_json_field(response, "emotion")

                    prompt = STRATEGY_TEMPLATE.format(context=context)
                    response = await self.call_model(TEXT_STRATEGY, prompt)
                    entry["strategy"] = self.extract_json_field(response, "strategy")
                else:
                    # Video-text emotion detection and strategy prediction for long therapist clips
                    prompt = VIDEO_TEXT_EMOTION_TEMPLATE.format(dialogue=dialogue)
                    response = await self.call_model(VIDEO_TEXT_EMOTION_DETECTION, prompt, video_path)
                    entry["emotion"] = self.extract_json_field(response, "emotion")

                    prompt = STRATEGY_TEMPLATE.format(context=context)
                    response = await self.call_model(VIDEO_TEXT_STRATEGY, prompt, video_path)
                    entry["strategy"] = self.extract_json_field(response, "strategy")

            return entry

        except Exception as e:
            self.logger.error(f"Error processing entry {video_path}: {str(e)}")
            return entry

    async def process_transcripts(self, input_file: str, output_file: str):
        """
        Process all transcripts in the input file
        
        Args:
            input_file (str): Path to input JSON file
            output_file (str): Path to output JSON file
        """
        try:
            # Read input JSON file
            self.logger.info(f"Reading input file: {input_file}")
            with open(input_file, 'r') as f:
                transcripts = json.load(f)

            # Process entries concurrently
            self.logger.info(f"Processing {len(transcripts)} transcript entries...")
            processed_entries = await asyncio.gather(
                *[self.process_entry(entry) for entry in transcripts]
            )

            # Save processed entries
            self.logger.info(f"Saving processed entries to: {output_file}")
            with open(output_file, 'w') as f:
                json.dump(processed_entries, f, indent=4)

            self.logger.info("Processing completed successfully")

        except Exception as e:
            self.logger.error(f"Error processing transcripts: {str(e)}")
            raise