import sys
import json
import torch
import random
import backoff
import logging
import asyncio
import vertexai
import colorlog
import transformers
from datetime import datetime
from dataclasses import dataclass
from google.api_core import exceptions
from ratelimit import limits, sleep_and_retry
from typing import Dict, Optional, List, Tuple
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

@dataclass
class TextProcessingItem:
    """
    Class for holding text-only processing items from transcripts.
    
    This dataclass organizes information needed for processing text-only items,
    particularly for short clips or corrupted videos that don't require video processing.
    
    Attributes:
        index (int): Original index in transcript for maintaining order
        role (str): Speaker role ('therapist' or 'client')
        dialogue (str): The actual transcript text to be analyzed
        context (str): Previous conversation context
        entry (Dict): Original transcript entry for updating with results
    """
    index: int  # Original index in transcript
    role: str
    dialogue: str
    context: str
    entry: Dict  # Original entry for updating

@dataclass
class VideoProcessingItem:
    """
    Class for holding video processing items from transcripts.
    
    This dataclass organizes information needed for processing video items,
    specifically for clips that require both video and text analysis.
    
    Attributes:
        index (int): Original index in transcript for maintaining order
        role (str): Speaker role ('therapist' or 'client')
        dialogue (str): The actual transcript text to be analyzed
        context (str): Previous conversation context
        video_path (str): Path to the video file for processing
        entry (Dict): Original transcript entry for updating with results
    """
    index: int  # Original index in transcript
    role: str
    dialogue: str
    context: str
    video_path: str
    entry: Dict  # Original entry for updating

class ModelError(Exception):
    """
    Custom exception class for model-related errors.
    
    This exception is raised when there are errors during model initialization,
    inference, or processing that are specific to the model operations.
    """
    pass

class TranscriptAnnotator:
    """
    Class for annotating therapy session transcripts using various models.
    
    This class manages the processing of therapy session transcripts using either
    local models (LLaMA and VideoLLaMA) or the Gemini API. It handles memory
    optimization, model initialization/cleanup, and transcript processing pipeline.
    
    Attributes:
        use_local (bool): Flag to use local models instead of Gemini API
        model_name (str): Name of the Gemini model to use
        logger (logging.Logger): Logger instance for this class
        llama_pipeline (Optional[Pipeline]): LLaMA model pipeline
        video_model (Optional[Model]): VideoLLaMA model instance
        processor (Optional[Processor]): Video processor for VideoLLaMA
        tokenizer (Optional[Tokenizer]): Tokenizer for VideoLLaMA
    """
    EMOTION_VALUES: List[str] = [
        "anger", "sadness", "disgust", "depression", 
        "neutral", "joy", "fear"
    ]
    
    STRATEGY_VALUES: List[str] = [
        "Open questions", "Approval", "Self-disclosure",
        "Restatement", "Interpretation", "Advisement",
        "Communication Skills", "Structuring the therapy",
        "Guiding the pace", "Others"
    ]


    def __init__(self, use_local: bool = False, model_name: str = "gemini-1.5-flash-002"):
        """
        Initialize the TranscriptAnnotator with specified configuration.
        
        Args:
            use_local (bool, optional): Whether to use local LLMs instead of Gemini API.
                Defaults to False.
            model_name (str, optional): Name of the Gemini model to use.
                Defaults to "gemini-1.5-flash-002".
                
        Note:
            When use_local is True, models are initialized lazily to optimize memory usage.
        """
        self.use_local = use_local
        self.model_name = model_name
        self.setup_logger()
        
        if not use_local:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            
        # Initialize models as None - they'll be loaded when needed
        self.llama_pipeline = None
        self.video_model = None
        self.processor = None
        self.tokenizer = None

    def setup_logger(self):
        """
        Configure colored logging for the TranscriptAnnotator.
        
        Sets up a logger with colored output formatting for different log levels
        and ensures no duplicate handlers exist.
        
        Returns:
            None
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

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicate logs
        for existing_handler in self.logger.handlers[:]:
            self.logger.removeHandler(existing_handler)
            
        self.logger.addHandler(handler)

    def _init_llama_model(self):
        """
        Initialize the LLaMA model for text processing.
        
        Loads the LLaMA model with optimized settings for text generation
        using bfloat16 precision and automatic device mapping.
        
        Raises:
            Exception: If there's an error during model initialization
            
        Note:
            This is called only when needed to optimize memory usage.
        """
        try:
            self.logger.info("Initializing LLaMA model...")
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            self.llama_pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        except Exception as e:
            self.logger.error(f"Error initializing LLaMA model: {str(e)}")
            raise

    def _init_videollama_model(self):
        """
        Initialize the VideoLLaMA model for video processing.
        
        Loads the VideoLLaMA model along with its processor and tokenizer
        for combined video-text analysis.
        
        Raises:
            Exception: If there's an error during model initialization
            
        Note:
            This is called only when needed to optimize memory usage.
        """
        try:
            self.logger.info("Initializing VideoLLaMA model...")
            model_path = "DAMO-NLP-SG/VideoLLaMA2-7B-16F"
            self.video_model, self.processor, self.tokenizer = model_init(model_path)
        except Exception as e:
            self.logger.error(f"Error initializing VideoLLaMA model: {str(e)}")
            raise

    def _cleanup_llama_model(self):
        """
        Clean up LLaMA model resources.
        
        Removes the LLaMA model from memory and clears CUDA cache
        to free up GPU memory for other models.
        """
        if self.llama_pipeline is not None:
            self.logger.info("Cleaning up LLaMA model resources...")
            del self.llama_pipeline
            self.llama_pipeline = None
            torch.cuda.empty_cache()

    def _cleanup_videollama_model(self):
        """
        Clean up VideoLLaMA model resources.
        
        Removes the VideoLLaMA model, processor, and tokenizer from memory
        and clears CUDA cache to free up GPU memory.
        """
        if self.video_model is not None:
            self.logger.info("Cleaning up VideoLLaMA model resources...")
            del self.video_model
            del self.processor
            del self.tokenizer
            self.video_model = None
            self.processor = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    async def _call_local_llama(self, system_instruction: str, prompt: str) -> str:
        """
        Call local LLaMA model for text processing with concatenated prompt.
        
        Args:
            system_instruction (str): System instruction for the model
            prompt (str): User prompt text for generation
            
        Returns:
            str: Generated model response text
            
        Raises:
            ModelError: If there's an error during model inference
        """
        try:
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ]
            
            outputs = self.llama_pipeline(
                messages,
                max_new_tokens=1024,
                temperature=0.1
            )
            
            full_response = outputs[0]['generated_text'][-1]['content']
            response = full_response.strip()
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling local LLaMA: {str(e)}")
            raise ModelError(f"LLaMA inference failed: {str(e)}")

    async def _call_local_videollama(self, video_path: str, system_instruction: str, prompt: str) -> str:
        """
        Call local VideoLLaMA model for video processing.
        
        Args:
            video_path (str): Path to the video file to process
            system_instruction (str): System instruction for the model
            prompt (str): User prompt for the model
            
        Returns:
            str: Generated model response text
            
        Raises:
            ModelError: If there's an error during model inference
        """
        try:
            # Preprocess video
            video_tensor = self.processor["video"](video_path)
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
            return ""

    @sleep_and_retry
    @limits(calls=CALLS_PER_MINUTE, period=60)
    @backoff.on_exception(
        backoff.expo,
        (exceptions.ServiceUnavailable, exceptions.InternalServerError),
        max_tries=MAX_RETRIES
    )
    async def _call_gemini(self, system_instruction: str, prompt: str, video_path: Optional[str] = None) -> str:
        """
        Call Gemini API with retries and rate limiting.
        
        Args:
            system_instruction (str): System instruction for the model
            prompt (str): Prompt template filled with data
            video_path (Optional[str]): Path to video file if needed
            
        Returns:
            str: Model response text
            
        Raises:
            exceptions.InvalidArgument: If the request is malformed
            exceptions.GoogleAPICallError: For general API errors
            
        Note:
            Includes automatic retries and rate limiting for API stability.
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

    async def call_model(self, system_instruction: str, prompt: str, video_path: Optional[str] = None) -> str:
        """
        Call appropriate model based on initialization configuration.
        
        This method routes the request to either local models or Gemini API
        based on the use_local flag set during initialization.
        
        Args:
            system_instruction (str): System instruction for the model
            prompt (str): Prompt template filled with data
            video_path (Optional[str]): Path to video file if needed
            
        Returns:
            str: Model response text
            
        Note:
            Automatically selects between LLaMA, VideoLLaMA, or Gemini API.
        """
        if self.use_local:
            if video_path:
                return await self._call_local_videollama(video_path, system_instruction, prompt)
            else:
                return await self._call_local_llama(system_instruction, prompt)
        else:
            return await self._call_gemini(system_instruction, prompt, video_path)

    def _get_fallback_value(self, field: str) -> str:
        """
        Get fallback value for a specific field when JSON extraction fails.
        
        Args:
            field (str): Field name to get fallback value for
            
        Returns:
            str: Appropriate fallback value for the field
        """
        fallback_map = {
            'emotion': random.choice(self.EMOTION_VALUES),
            'strategy': random.choice(self.STRATEGY_VALUES),
            'emotional_cues': ""
        }
        return fallback_map.get(field, "")

    def _create_error_message(self, video_path: Optional[str], field: str, 
                            error: str, response_text: str) -> Dict:
        """
        Create error message dictionary for logging.
        
        Args:
            video_path (Optional[str]): Path to video file
            field (str): Field that failed extraction
            error (str): Error description
            response_text (str): Original response text
            
        Returns:
            Dict: Formatted error message
        """
        return {
            'video_path': video_path or 'unknown_path',
            'field': field,
            'error': error,
            'response': response_text[:100] + '...' if len(response_text) > 100 else response_text
        }

    def _extract_json_str(self, response_text: str) -> Optional[str]:
        """
        Extract JSON string from response text.
        
        Args:
            response_text (str): Text containing JSON
            
        Returns:
            Optional[str]: Extracted JSON string or None if not found
        """
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            return None
            
        return response_text[start_idx:end_idx]

    def extract_json_field(self, response_text: str, field: str, video_path: Optional[str] = None) -> str:
        """
        Extract specific field from JSON response text with fallback values.
        
        Args:
            response_text (str): Response text containing JSON
            field (str): Field to extract from JSON
            video_path (Optional[str]): Path to video file for logging purposes
            
        Returns:
            str: Extracted field value or fallback value if JSON not found
        """
        try:
            json_str = self._extract_json_str(response_text)
            if not json_str:
                error_msg = self._create_error_message(
                    video_path, field, 'No JSON found in response', response_text
                )
                self._log_json_error(error_msg)
                return self._get_fallback_value(field)
                
            return json.loads(json_str)[field]
            
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = self._create_error_message(
                video_path, field, str(e), response_text
            )
            self._log_json_error(error_msg)
            return self._get_fallback_value(field)

    def _log_json_error(self, error_msg: Dict):
        """
        Log JSON extraction errors to a file.
        
        Args:
            error_msg (Dict): Dictionary containing error details
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = {
                'timestamp': timestamp,
                **error_msg
            }
            
            with open('json_extraction_errors.json', 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
                
            self.logger.warning(
                f"JSON extraction failed for {error_msg['field']} "
                f"in {error_msg['video_path']}"
            )
            
        except Exception as e:
            self.logger.error(f"Error writing to error log: {str(e)}")

    def _sort_processing_items(self, transcripts: List[Dict]) -> Tuple[List[TextProcessingItem], List[VideoProcessingItem]]:
        """
        Sort transcript entries into text-only and video processing items.
        
        Organizes transcript entries into two queues based on whether they
        need text-only or video processing.
        
        Args:
            transcripts (List[Dict]): List of transcript entries to sort
            
        Returns:
            Tuple[List[TextProcessingItem], List[VideoProcessingItem]]: 
                Two lists containing sorted processing items:
                - First list contains items needing text-only processing
                - Second list contains items needing video processing
                
        Note:
            Sorting is based on clip duration and corruption status.
        """
        text_items = []
        video_items = []
        
        for idx, entry in enumerate(transcripts):
            role = entry["role"]
            is_short = entry["short_clip"] == "Yes"
            is_corrupted = entry["corrupted"] == "Yes"
            dialogue = entry["transcript"]
            context = entry["previous_transcript"]
            
            if is_short or is_corrupted:
                text_items.append(TextProcessingItem(
                    index=idx,
                    role=role,
                    dialogue=dialogue,
                    context=context,
                    entry=entry
                ))
            else:
                video_items.append(VideoProcessingItem(
                    index=idx,
                    role=role,
                    dialogue=dialogue,
                    context=context,
                    video_path=entry["path"],
                    entry=entry
                ))
                
        return text_items, video_items

    async def _process_text_item(self, item: TextProcessingItem) -> None:
        """
        Process a text-only item using LLaMA model.
        
        Handles emotion detection for all roles and strategy prediction
        for therapist roles using text-only processing.
        
        Args:
            item (TextProcessingItem): Item containing text to process
            
        Note:
            Updates the original entry in place with processing results.
        """
        try:
            # Emotion detection
            prompt = TEXT_EMOTION_TEMPLATE.format(role=item.role, dialogue=item.dialogue)
            response = await self._call_local_llama(TEXT_EMOTION_DETECTION, prompt)
            item.entry["emotion"] = self.extract_json_field(response, "emotion", item.entry["path"])
            
            # Strategy prediction for therapist
            if item.role == "therapist":
                prompt = STRATEGY_TEMPLATE.format(context=item.context)
                response = await self._call_local_llama(TEXT_STRATEGY, prompt)
                item.entry["strategy"] = self.extract_json_field(response, "strategy", item.entry["path"])
                
        except Exception as e:
            self.logger.error(f"Error processing text item: {str(e)}")

    async def _process_video_item(self, item: VideoProcessingItem) -> None:
        """
        Process a video item using VideoLLaMA model.
        
        Handles emotion detection and analysis for all roles, plus strategy
        prediction for therapist roles using video-text processing.
        
        Args:
            item (VideoProcessingItem): Item containing video to process
            
        Note:
            Updates the original entry in place with processing results.
        """

        try:
            # Emotion detection
            prompt = VIDEO_TEXT_EMOTION_TEMPLATE.format(dialogue=item.dialogue)
            response = await self._call_local_videollama(
                item.video_path, 
                VIDEO_TEXT_EMOTION_DETECTION, 
                prompt
            )
            item.entry["emotion"] = self.extract_json_field(response, "emotion", item.entry["path"])
            
            if item.role == "client":
                # Analysis for client
                response = await self._call_local_videollama(
                    item.video_path,
                    VIDEO_TEXT_EMOTION_ANALYSIS,
                    VIDEO_TEXT_ANALYSIS_TEMPLATE
                )
                item.entry["analysis"] = response
            else:
                # Strategy prediction for therapist
                prompt = STRATEGY_TEMPLATE.format(context=item.context)
                response = await self._call_local_videollama(
                    item.video_path,
                    VIDEO_TEXT_STRATEGY,
                    prompt
                )
                item.entry["strategy"] = self.extract_json_field(response, "strategy", item.entry["path"])
                
        except Exception as e:
            self.logger.error(f"Error processing video item: {str(e)}")

    async def _process_video_item_sequential(self, item: VideoProcessingItem) -> None:
        """
        Process a video item using VideoLLaMA model sequentially.
        
        Handles emotion detection and analysis for all roles, plus strategy
        prediction for therapist roles using video-text processing. Each
        VideoLLaMA call is made sequentially to prevent memory issues.
        
        Args:
            item (VideoProcessingItem): Item containing video to process
            
        Note:
            Updates the original entry in place with processing results.
            This method processes each VideoLLaMA call one at a time to
            ensure stable memory usage.
        """
        try:
            # Emotion detection
            prompt = VIDEO_TEXT_EMOTION_TEMPLATE.format(dialogue=item.dialogue)
            response = await self._call_local_videollama(
                item.video_path, 
                VIDEO_TEXT_EMOTION_DETECTION, 
                prompt
            )
            item.entry["emotion"] = self.extract_json_field(response, "emotion", item.video_path)
            
            if item.role == "client":
                # Analysis for client
                response = await self._call_local_videollama(
                    item.video_path,
                    VIDEO_TEXT_EMOTION_ANALYSIS,
                    VIDEO_TEXT_ANALYSIS_TEMPLATE
                )
                item.entry["analysis"] = response
            else:
                # Strategy prediction for therapist
                prompt = STRATEGY_TEMPLATE.format(context=item.context)
                response = await self._call_local_videollama(
                    item.video_path,
                    VIDEO_TEXT_STRATEGY,
                    prompt
                )
                item.entry["strategy"] = self.extract_json_field(response, "strategy", item.video_path)
                
        except Exception as e:
            self.logger.error(f"Error processing video item {item.video_path}: {str(e)}")

    async def process_entry(self, entry: Dict) -> Dict:
        """
        Process a single transcript entry using Gemini API.
        
        Handles complete processing of a transcript entry including emotion
        detection, analysis, and strategy prediction as needed.
        
        Args:
            entry (Dict): Transcript entry to process
            
        Returns:
            Dict: Processed entry with model predictions
            
        Note:
            This method is primarily used for Gemini API processing.
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
                    entry["emotion"] = self.extract_json_field(response, "emotion", video_path)
                else:
                    # Video-text emotion detection and analysis for long client clips
                    prompt = VIDEO_TEXT_EMOTION_TEMPLATE.format(dialogue=dialogue)
                    response = await self.call_model(VIDEO_TEXT_EMOTION_DETECTION, prompt, video_path)
                    entry["emotion"] = self.extract_json_field(response, "emotion", video_path)

                    response = await self.call_model(VIDEO_TEXT_EMOTION_ANALYSIS, VIDEO_TEXT_ANALYSIS_TEMPLATE, video_path)
                    entry["analysis"] = response

            # Therapist clips
            else:
                if is_short or is_corrupted:
                    # Text emotion detection and strategy prediction for short therapist clips
                    prompt = TEXT_EMOTION_TEMPLATE.format(role=role, dialogue=dialogue)
                    response = await self.call_model(TEXT_EMOTION_DETECTION, prompt)
                    entry["emotion"] = self.extract_json_field(response, "emotion", video_path)

                    prompt = STRATEGY_TEMPLATE.format(context=context)
                    response = await self.call_model(TEXT_STRATEGY, prompt)
                    entry["strategy"] = self.extract_json_field(response, "strategy", video_path)
                else:
                    # Video-text emotion detection and strategy prediction for long therapist clips
                    prompt = VIDEO_TEXT_EMOTION_TEMPLATE.format(dialogue=dialogue)
                    response = await self.call_model(VIDEO_TEXT_EMOTION_DETECTION, prompt, video_path)
                    entry["emotion"] = self.extract_json_field(response, "emotion", video_path)

                    prompt = STRATEGY_TEMPLATE.format(context=context)
                    response = await self.call_model(VIDEO_TEXT_STRATEGY, prompt, video_path)
                    entry["strategy"] = self.extract_json_field(response, "strategy", video_path)

            return entry

        except Exception as e:
            self.logger.error(f"Error processing entry {video_path}: {str(e)}")
            return entry

    async def process_transcripts(self, input_file: str, output_file: str):
        """
        Process all transcripts with optimized memory usage.
        
        Main processing pipeline that handles reading input, processing all
        entries with appropriate models, and saving results. VideoLLaMA 
        processing is done sequentially to prevent memory issues, while
        LLaMA processing remains asynchronous for better performance.
        
        Args:
            input_file (str): Path to input JSON file
            output_file (str): Path to output JSON file
            
        Note:
            - VideoLLaMA processing is sequential to prevent memory issues
            - LLaMA processing remains asynchronous for better throughput
            - For Gemini API, processes all entries in parallel
        
        Raises:
            Exception: If there's an error during processing
        """
        try:
            # Read input JSON file
            self.logger.info(f"Reading input file: {input_file}")
            with open(input_file, 'r') as f:
                transcripts = json.load(f)

            if self.use_local:
                # Sort items into text and video processing queues
                text_items, video_items = self._sort_processing_items(transcripts)
                self.logger.info(f"Sorted {len(text_items)} text items and {len(video_items)} video items")

                # Process text items with LLaMA (keep async for better throughput)
                if text_items:
                    self.logger.info("Processing text-only items with LLaMA...")
                    self._init_llama_model()
                    await asyncio.gather(*[self._process_text_item(item) for item in text_items])
                    self._cleanup_llama_model()

                # Process video items sequentially with VideoLLaMA
                if video_items:
                    self.logger.info("Processing video items with VideoLLaMA sequentially...")
                    self._init_videollama_model()
                    for item in video_items:
                        self.logger.debug(f"Processing video item: {item.video_path}")
                        await self._process_video_item_sequential(item)
                    self._cleanup_videollama_model()
                
            else:
                # Use Gemini API and update transcripts with processed results
                self.logger.info(f"Processing {len(transcripts)} entries using Gemini API...")
                processed_entries = await asyncio.gather(
                    *[self.process_entry(entry) for entry in transcripts]
                )
                transcripts = processed_entries

            # Save processed entries
            self.logger.info(f"Saving processed entries to: {output_file}")
            with open(output_file, 'w') as f:
                json.dump(transcripts, f, indent=4)

            self.logger.info("Processing completed successfully")

        except Exception as e:
            self.logger.error(f"Error processing transcripts: {str(e)}")
            raise