"""
Core speech diarization processor module.

This module contains the main SpeechDiarizationProcessor class that handles
audio transcription with speaker diarization and word-level timestamps.
"""

import csv
import json
import logging
import os
import sys
import time
import math
from typing import List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from .models import DiaryMetadata

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed. Run: pip install faster-whisper")
    sys.exit(1)

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation, Segment
except ImportError:
    print("Error: pyannote.audio not installed. Run: pip install pyannote.audio")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)


class SpeechDiarizationProcessor:
    """
    A class for processing audio files with speaker diarization and word-level timestamps.
    """
    
    def __init__(self, whisper_model_size: str = "large-v3", device: str = "auto", 
                 compute_type: str = "float16", hf_token: Optional[str] = None,
                 num_speakers: Optional[int] = None, min_speakers: int = 1, 
                 max_speakers: Optional[int] = None):
        """
        Initialize the processor with models.
        
        Args:
            whisper_model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large-v3', etc.)
            device: Device to use ('cpu', 'cuda', 'auto')
            compute_type: Compute type for Whisper ('float16', 'int8', 'float32')
            hf_token: HuggingFace access token for pyannote models
            num_speakers: Exact number of speakers (improves accuracy when known)
            min_speakers: Minimum number of speakers (default: 1)
            max_speakers: Maximum number of speakers
        """
        self.device = self._setup_device(device)
        self.hf_token = hf_token
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        
        logger.info(f"Initializing models on device: {self.device}")
        
        # Initialize Whisper model
        try:
            logger.info(f"Loading Whisper model: {whisper_model_size}")
            self.whisper_model = WhisperModel(
                whisper_model_size, 
                device=self.device, 
                compute_type=compute_type
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
        
        # Initialize pyannote pipeline
        try:
            logger.info("Loading pyannote speaker diarization pipeline...")
            if not hf_token:
                raise ValueError("HuggingFace token is required for pyannote models")
            
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # Move to GPU if available (stays on CPU otherwise)
            if self.device == "cuda" and torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
                logger.info("Moved diarization pipeline to GPU")
            else:
                logger.info("Using CPU for diarization pipeline")
            
            logger.info("Pyannote pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load pyannote pipeline: {e}")
            logger.error("Make sure you have:")
            logger.error("1. Accepted user conditions at https://hf.co/pyannote/speaker-diarization-3.1")
            logger.error("2. Valid HuggingFace access token")
            raise
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return device
    
    SUPPORTED_FORMATS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm')

    def validate_audio_file(self, audio_path: str) -> bool:
        """
        Validate that the audio file exists and is a supported format.

        Args:
            audio_path: Path to the audio file

        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False

        if not audio_path.lower().endswith(self.SUPPORTED_FORMATS):
            logger.error(f"Unsupported format. Supported: {self.SUPPORTED_FORMATS}")
            return False
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            return False
        
        logger.info(f"Audio file validated: {audio_path} ({file_size / 1024 / 1024:.1f} MB)")
        return True
    
    def transcribe_audio(self, audio_path: str) -> Tuple[List[dict], dict]:
        """
        Transcribe audio using faster-whisper with word-level timestamps.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (segments_list, transcription_info)
        """
        logger.info("Starting transcription with Whisper...")
        start_time = time.time()
        
        try:
            # Transcribe with word-level timestamps
            segments, info = self.whisper_model.transcribe(
                audio_path,
                word_timestamps=True,
                beam_size=5,
                vad_filter=True,  # Voice activity detection
                temperature=0.0   # Deterministic output
            )
            
            # Convert generator to list
            segments_list = list(segments)
            
            transcription_time = time.time() - start_time
            logger.info(f"Transcription completed in {transcription_time:.2f} seconds")
            logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            logger.info(f"Total segments: {len(segments_list)}")
            
            # Convert transcription info to dictionary
            info_dict = {
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': info.duration,
                'duration_after_vad': getattr(info, 'duration_after_vad', info.duration)
            }
            
            return segments_list, info_dict
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def perform_speaker_diarization(self, audio_path: str) -> Annotation:
        """
        Perform speaker diarization using pyannote.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Pyannote Annotation object with speaker segments
        """
        logger.info("Starting speaker diarization...")
        start_time = time.time()
        
        try:
            # Prepare diarization parameters
            diarization_params = {}
            
            if self.num_speakers is not None:
                # If exact number is specified, use it
                diarization_params['num_speakers'] = self.num_speakers
                logger.info(f"Using fixed number of speakers: {self.num_speakers}")
            else:
                # Use min/max range if specified
                if self.min_speakers > 1:
                    diarization_params['min_speakers'] = self.min_speakers
                    logger.info(f"Minimum speakers: {self.min_speakers}")
                if self.max_speakers is not None:
                    diarization_params['max_speakers'] = self.max_speakers
                    logger.info(f"Maximum speakers: {self.max_speakers}")
            
            # Run diarization pipeline with parameters
            if diarization_params:
                diarization = self.diarization_pipeline(audio_path, **diarization_params)
            else:
                diarization = self.diarization_pipeline(audio_path)
            
            diarization_time = time.time() - start_time
            num_speakers = len(diarization.labels())
            logger.info(f"Diarization completed in {diarization_time:.2f} seconds")
            logger.info(f"Detected {num_speakers} speakers: {sorted(diarization.labels())}")
            
            return diarization
            
        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            raise
    
    def align_transcription_with_speakers(self, segments: List[dict], 
                                        diarization: Annotation) -> List[dict]:
        """
        Align transcribed segments with speaker diarization results.
        
        Args:
            segments: List of transcription segments from Whisper
            diarization: Speaker diarization results from pyannote
            
        Returns:
            List of aligned segments with speaker assignments
        """
        logger.info("Aligning transcription with speaker diarization...")
        
        aligned_segments = []
        
        for segment in tqdm(segments, desc="Aligning segments"):
            segment_start = segment.start
            segment_end = segment.end
            segment_text = segment.text.strip()
            
            # Create segment for overlap calculation
            segment_interval = Segment(segment_start, segment_end)
            
            # Find the speaker with maximum overlap
            best_speaker = None
            max_overlap = 0.0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap = (turn & segment_interval).duration
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker
            
            # If no overlap found, assign to unknown speaker
            if best_speaker is None:
                best_speaker = "SPEAKER_UNKNOWN"
            
            # Process word-level timestamps
            words_list = []
            if hasattr(segment, 'words') and segment.words:
                for word_idx, word in enumerate(segment.words):
                    word_dict = {
                        "word": word.word.strip(),
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "confidence": getattr(word, 'probability', 1.0),
                        "position": word_idx
                    }
                    words_list.append(word_dict)
            
            # Create aligned segment
            aligned_segment = {
                "speaker": best_speaker,
                "start": round(segment_start, 3),
                "end": round(segment_end, 3),
                "text": segment_text,
                "words": words_list,
                "segment_confidence": math.exp(getattr(segment, 'avg_logprob', 0.0)),
                "overlap_duration": round(max_overlap, 3)
            }
            
            aligned_segments.append(aligned_segment)
        
        logger.info(f"Alignment completed. {len(aligned_segments)} segments processed.")
        return aligned_segments
    
    def save_to_csv(self, aligned_segments: List[dict], output_path: str, 
                   transcription_info: dict) -> None:
        """
        Save aligned segments to CSV file with word-level details.
        
        Args:
            aligned_segments: List of aligned segments
            output_path: Path to output CSV file
            transcription_info: Metadata from transcription
        """
        logger.info(f"Saving results to CSV: {output_path}")
        
        # Prepare rows for CSV
        csv_rows = []
        
        for segment in aligned_segments:
            # Add segment-level row
            segment_row = {
                "type": "segment",
                "speaker": segment["speaker"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "word": "",
                "word_position": "",
                "confidence": segment.get("segment_confidence", 1.0),
                "overlap_duration": segment.get("overlap_duration", 0.0)
            }
            csv_rows.append(segment_row)
            
            # Add word-level rows
            for word in segment["words"]:
                word_row = {
                    "type": "word",
                    "speaker": segment["speaker"],
                    "start": word["start"],
                    "end": word["end"],
                    "text": segment["text"],  # Keep full segment text for reference
                    "word": word["word"],
                    "word_position": word["position"],
                    "confidence": word["confidence"],
                    "overlap_duration": segment.get("overlap_duration", 0.0)
                }
                csv_rows.append(word_row)
        
        # Save to CSV
        try:
            df = pd.DataFrame(csv_rows)
            df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
            logger.info(f"CSV saved successfully with {len(csv_rows)} rows")
            
            # Save metadata to companion JSON file
            metadata_path = output_path.replace('.csv', '_metadata.json')
            metadata = {
                "audio_file": transcription_info.get("audio_file", "unknown"),
                "language": transcription_info.get("language", "unknown"),
                "language_probability": transcription_info.get("language_probability", 0.0),
                "duration": transcription_info.get("duration", 0.0),
                "total_segments": len(aligned_segments),
                "total_words": sum(len(seg["words"]) for seg in aligned_segments),
                "speakers": list(set(seg["speaker"] for seg in aligned_segments)),
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Metadata saved to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            raise
    
    def process_audio_file(self, audio_path: str, output_path: str) -> None:
        """
        Complete processing pipeline: transcription + diarization + alignment + save.
        
        Args:
            audio_path: Path to input .wav file
            output_path: Path to output .csv file
        """
        logger.info(f"Starting processing of: {audio_path}")
        start_time = time.time()
        
        try:
            # Validate input
            if not self.validate_audio_file(audio_path):
                raise ValueError(f"Invalid audio file: {audio_path}")
            
            # Step 1: Transcribe audio
            segments, transcription_info = self.transcribe_audio(audio_path)
            transcription_info["audio_file"] = audio_path
            
            # Step 2: Perform speaker diarization
            diarization = self.perform_speaker_diarization(audio_path)
            
            # Step 3: Align transcription with speakers
            aligned_segments = self.align_transcription_with_speakers(segments, diarization)
            
            # Step 4: Save results
            self.save_to_csv(aligned_segments, output_path, transcription_info)
            
            total_time = time.time() - start_time
            logger.info(f"Processing completed successfully in {total_time:.2f} seconds")
            logger.info(f"Output saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
