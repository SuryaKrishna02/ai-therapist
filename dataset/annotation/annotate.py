import json
import backoff
import logging
import asyncio
import vertexai
import colorlog
import sys
import torch
import transformers
from typing import Dict, Optional
from google.api_core import exceptions
from ratelimit import limits, sleep_and_retry
from vertexai.generative_models import GenerativeModel, Part
sys.path.append('./././')
from videollama2 import model_init, mm_infer

# Import existing constants
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
    
    def __init__(self, use_local: bool = False, model_name: str = "gemini-1.5-flash-002"):
        """
        Initialize the TranscriptAnnotator
        
        Args:
            use_local (bool): Whether to use local LLMs instead of Gemini API
            model_name (str): Name of the generative model to use for Gemini
        """
        self.use_local = use_local
        self.model_name = model_name
        self.setup_logger()
        
        if not use_local:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
        else:
            # Initialize local models
            self._init_local_models()

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
        
        # Remove any existing handlers to avoid duplicate logs
        for existing_handler in self.logger.handlers[:]:
            self.logger.removeHandler(existing_handler)
            
        self.logger.addHandler(handler)

    def _init_local_models(self):
        """Initialize local LLaMA and VideoLLaMA models"""
        try:
            # Initialize LLaMA model for text
            self.logger.info("Initializing LLaMA model...")
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            self.llama_pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
            
            # Initialize VideoLLaMA model
            self.logger.info("Initializing VideoLLaMA model...")
            model_path = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
            self.video_model, self.processor, self.tokenizer = model_init(model_path)
            
        except Exception as e:
            self.logger.error(f"Error initializing local models: {str(e)}")
            raise

    async def _call_local_llama(self, system_instruction: str, prompt: str) -> str:
        """
        Call local LLaMA model for text processing with concatenated prompt
        
        Args:
            system_instruction (str): System instruction for the model
            prompt (str): User prompt
            
        Returns:
            str: Model response text
        """
        try:
            # Concatenate system instruction and prompt with newline
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ]
            
            outputs = self.llama_pipeline(
                messages,
                max_new_tokens=1024,
                temperature=0.1
            )
            
            # Extract the generated response
            full_response = outputs[0]['generated_text'][-1]['content']
            # Return only the newly generated text (after the prompt)
            response = full_response.strip()
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling local LLaMA: {str(e)}")
            raise ModelError(f"LLaMA inference failed: {str(e)}")

    async def _call_local_videollama(self, video_path: str, system_instruction:str, prompt: str) -> str:
        """
        Call local VideoLLaMA model for video processing
        
        Args:
            video_path (str): Path to video file
            prompt (str): Question/prompt for the model
            
        Returns:
            str: Model response text
        """
        try:
            # Preprocess video
            preprocess = self.processor["video"]
            video_tensor = preprocess(video_path, va=True)

            combined_prompt = f"{system_instruction}\n{prompt}"
            
            # Run inference
            output = mm_infer(
                video_tensor,
                combined_prompt,
                model=self.video_model,
                tokenizer=self.tokenizer,
                modal="video",
                do_sample=False,
            )
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error calling VideoLLaMA: {str(e)}")
            raise ModelError(f"VideoLLaMA inference failed: {str(e)}")

    @sleep_and_retry
    @limits(calls=CALLS_PER_MINUTE, period=60)
    @backoff.on_exception(
        backoff.expo,
        (exceptions.ServiceUnavailable, exceptions.InternalServerError),
        max_tries=MAX_RETRIES
    )
    async def _call_gemini(self, 
                          system_instruction: str,
                          prompt: str,
                          video_path: Optional[str] = None) -> str:
        """
        Call Gemini API with retries and rate limiting
        
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
            self.logger.error(f"Bad request for video {video_path}: {str(e)}")
            with open('corrupted_files.txt', 'a') as f:
                f.write(f"{video_path}\n")
            raise

        except (exceptions.ServiceUnavailable, exceptions.InternalServerError) as e:
            self.logger.warning(f"Server error, retrying: {str(e)}")
            raise

        except exceptions.GoogleAPICallError as e:
            self.logger.error(f"API error for video {video_path}: {str(e)}")
            raise

    async def call_model(self, 
                        system_instruction: str,
                        prompt: str,
                        video_path: Optional[str] = None) -> str:
        """
        Call appropriate model based on initialization flag
        
        Args:
            system_instruction (str): System instruction for the model
            prompt (str): Prompt template filled with data
            video_path (Optional[str]): Path to video file if needed
            
        Returns:
            str: Model response text
        """
        if self.use_local:
            if video_path:
                return await self._call_local_videollama(video_path, system_instruction, prompt)
            else:
                return await self._call_local_llama(system_instruction, prompt)
        else:
            return await self._call_gemini(system_instruction, prompt, video_path)

    def extract_json_field(self, response_text: str, field: str) -> str:
        """
        Extract specific field from JSON response text
        
        Args:
            response_text (str): Response text containing JSON
            field (str): Field to extract from JSON
            
        Returns:
            str: Extracted field value
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
            is_corrupted = entry["corrupted"] == "Yes"

            # Client clips
            if role == "client":
                if is_short or is_corrupted:
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
                if is_short or is_corrupted:
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