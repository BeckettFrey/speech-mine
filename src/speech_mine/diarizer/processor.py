"""
Core speech diarization processor module.

Uses WhisperX for transcription, forced word-level alignment, and speaker diarization.
"""

import csv
import gc
import json
import logging
import math
import os
import time
from typing import Optional

import pandas as pd
import torch

logger = logging.getLogger(__name__)


class SpeechDiarizationProcessor:
    """
    Processes audio files using WhisperX: transcription, forced alignment, and speaker diarization.
    Produces the same dual-row CSV format as before (segment + word rows).
    """

    SUPPORTED_FORMATS = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.webm')

    def __init__(
        self,
        whisper_model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
        hf_token: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: Optional[int] = None,
        batch_size: int = 16,
        language: Optional[str] = None,
    ):
        """
        Args:
            whisper_model_size: Whisper model variant (tiny/base/small/medium/large-v2/large-v3/turbo).
            device: 'auto', 'cpu', or 'cuda'.
            compute_type: 'float16', 'int8', or 'float32'.
            hf_token: HuggingFace token (required for speaker diarization via pyannote).
            num_speakers: Exact speaker count if known.
            min_speakers: Minimum speakers (default 1).
            max_speakers: Maximum speakers.
            batch_size: Batch size for WhisperX transcription (reduce if OOM).
            language: Language code (e.g. 'en'). Auto-detected if None.
        """
        try:
            import whisperx
        except ImportError:
            raise ImportError("whisperx is not installed. Run: pip install whisperx")

        self.device = self._setup_device(device)
        self.hf_token = hf_token
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.batch_size = batch_size
        self.language = language

        logger.info(f"Loading WhisperX model '{whisper_model_size}' on {self.device}")
        self.model = whisperx.load_model(
            whisper_model_size,
            self.device,
            compute_type=compute_type,
            language=language,
        )
        logger.info("WhisperX model loaded")

    def _setup_device(self, device: str) -> str:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        return device

    def validate_audio_file(self, audio_path: str) -> bool:
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False
        if not audio_path.lower().endswith(self.SUPPORTED_FORMATS):
            logger.error(f"Unsupported format. Supported: {self.SUPPORTED_FORMATS}")
            return False
        if os.path.getsize(audio_path) == 0:
            logger.error(f"Audio file is empty: {audio_path}")
            return False
        return True

    def transcribe_audio(self, audio_path: str):
        """Transcribe audio. Returns (audio_array, result_dict)."""
        import whisperx

        logger.info("Loading audio...")
        audio = whisperx.load_audio(audio_path)
        duration = round(len(audio) / 16000, 3)

        logger.info("Transcribing with WhisperX...")
        start = time.time()
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        logger.info(f"Transcription done in {time.time() - start:.2f}s — language: {result.get('language')}")

        result["duration"] = duration
        # WhisperX surfaces language_probability via the underlying faster-whisper info
        # when it runs detect_language; fall back to 0.0 if unavailable.
        if "language_probability" not in result:
            try:
                _lang, prob, _all = self.model.model.detect_language(audio)
                result["language_probability"] = round(prob, 4)
            except Exception:
                result["language_probability"] = 0.0

        return audio, result

    def align(self, audio, result: dict) -> dict:
        """Forced word-level alignment using wav2vec2."""
        import whisperx

        language = result.get("language", self.language or "en")
        logger.info(f"Loading alignment model for language '{language}'...")
        align_model, metadata = whisperx.load_align_model(language_code=language, device=self.device)

        logger.info("Running forced alignment...")
        result = whisperx.align(result["segments"], align_model, metadata, audio, self.device)

        del align_model
        gc.collect()
        return result

    def diarize(self, audio, result: dict) -> dict:
        """Speaker diarization via pyannote, then assign speakers to words."""
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        if not self.hf_token:
            raise ValueError("HuggingFace token required for speaker diarization")

        logger.info("Loading diarization pipeline...")
        diarize_model = DiarizationPipeline(token=self.hf_token, device=self.device)

        params = {}
        if self.num_speakers is not None:
            params["num_speakers"] = self.num_speakers
        else:
            if self.min_speakers > 1:
                params["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                params["max_speakers"] = self.max_speakers

        logger.info("Running speaker diarization...")
        diarize_segments = diarize_model(audio, **params)

        logger.info("Assigning speakers to words...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        del diarize_model
        gc.collect()
        return result

    def save_to_csv(self, result: dict, output_path: str, info: dict) -> None:
        """Write aligned + diarized result to the dual-row CSV format."""
        segments = result.get("segments", [])
        csv_rows = []

        for segment in segments:
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            seg_confidence = math.exp(segment.get("avg_logprob", 0.0))

            csv_rows.append({
                "type": "segment",
                "speaker": speaker,
                "start": round(segment["start"], 3),
                "end": round(segment["end"], 3),
                "text": segment["text"].strip(),
                "word": "",
                "word_position": "",
                "confidence": round(seg_confidence, 6),
                "overlap_duration": 0.0,
            })

            for idx, word in enumerate(segment.get("words", [])):
                word_speaker = word.get("speaker", speaker)
                csv_rows.append({
                    "type": "word",
                    "speaker": word_speaker,
                    "start": round(word.get("start", segment["start"]), 3),
                    "end": round(word.get("end", segment["end"]), 3),
                    "text": segment["text"].strip(),
                    "word": word["word"].strip(),
                    "word_position": idx,
                    "confidence": round(word.get("score", 1.0), 6),
                    "overlap_duration": 0.0,
                })

        df = pd.DataFrame(csv_rows)
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"CSV saved: {output_path} ({len(csv_rows)} rows)")

        metadata_path = output_path.replace(".csv", "_metadata.json")
        speakers = sorted({r["speaker"] for r in csv_rows if r["type"] == "segment"})
        metadata = {
            "audio_file": info.get("audio_file", "unknown"),
            "language": info.get("language", "unknown"),
            "language_probability": info.get("language_probability", 0.0),
            "duration": info.get("duration", 0.0),
            "total_segments": len(segments),
            "total_words": sum(len(s.get("words", [])) for s in segments),
            "speakers": speakers,
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata saved: {metadata_path}")

    def process_audio_file(self, audio_path: str, output_path: str) -> None:
        """Full pipeline: transcribe → align → diarize → save CSV."""
        if not self.validate_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")

        start = time.time()

        audio, result = self.transcribe_audio(audio_path)
        info = {
            "audio_file": audio_path,
            "language": result.get("language", "unknown"),
            "language_probability": result.get("language_probability", 0.0),
            "duration": result.get("duration", 0.0),
        }

        result = self.align(audio, result)
        result = self.diarize(audio, result)

        self.save_to_csv(result, output_path, info)
        logger.info(f"Processing complete in {time.time() - start:.2f}s → {output_path}")
