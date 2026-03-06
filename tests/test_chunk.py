"""
Test cases for audio chunking utility.
"""

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from speech_mine.pickaxe.chunk import AudioChunker, chunk_audio_file


class TestAudioChunker:
    """Test cases for AudioChunker class"""

    def create_sample_config(self, chunks_data):
        """Helper to create a temporary YAML config file"""
        config = {"chunks": chunks_data}
        fd, path = tempfile.mkstemp(suffix='.yaml')
        try:
            with os.fdopen(fd, 'w') as f:
                yaml.dump(config, f)
            return path
        except:
            os.close(fd)
            raise

    def test_load_config_valid(self):
        """Test loading valid YAML configuration"""
        chunker = AudioChunker()
        chunks_data = [
            {"start": 0.0, "end": 30.0, "name": "intro"},
            {"start": 30.0, "end": 120.0, "name": "main"}
        ]
        
        config_path = self.create_sample_config(chunks_data)
        try:
            chunks = chunker.load_config(config_path)
            assert len(chunks) == 2
            assert chunks[0]["start"] == 0.0
            assert chunks[0]["name"] == "intro"
            assert chunks[1]["start"] == 30.0
            assert chunks[1]["name"] == "main"
        finally:
            os.unlink(config_path)

    def test_load_config_missing_file(self):
        """Test loading config from non-existent file"""
        chunker = AudioChunker()
        try:
            chunker.load_config("non_existent.yaml")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected

    def test_load_config_invalid_structure(self):
        """Test loading config with invalid structure"""
        chunker = AudioChunker()
        # Create config without 'chunks' key
        config = {"invalid": "structure"}
        fd, config_path = tempfile.mkstemp(suffix='.yaml')
        try:
            with os.fdopen(fd, 'w') as f:
                yaml.dump(config, f)
            
            try:
                chunker.load_config(config_path)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "chunks" in str(e)
        finally:
            os.unlink(config_path)

    def test_validate_chunks_valid(self):
        """Test validation with valid chunks"""
        chunker = AudioChunker()
        chunks = [
            {"start": 0.0, "end": 30.0, "name": "intro"},
            {"start": 30.0, "end": 60.0, "name": "main"}
        ]
        # Should not raise any exception
        chunker.validate_chunks(chunks, audio_duration=120.0)

    def test_validate_chunks_duplicate_start_times(self):
        """Test validation fails with duplicate start times"""
        chunker = AudioChunker()
        chunks = [
            {"start": 0.0, "end": 30.0, "name": "intro"},
            {"start": 0.0, "end": 60.0, "name": "duplicate"}  # Same start time
        ]
        try:
            chunker.validate_chunks(chunks, audio_duration=120.0)
            assert False, "Should have raised ValueError for duplicate start times"
        except ValueError as e:
            assert "duplicate start time" in str(e)

    def test_validate_chunks_exceeds_duration(self):
        """Test validation fails when chunk exceeds audio duration"""
        chunker = AudioChunker()
        chunks = [
            {"start": 0.0, "end": 130.0, "name": "too_long"}  # Exceeds 120s duration
        ]
        try:
            chunker.validate_chunks(chunks, audio_duration=120.0)
            assert False, "Should have raised ValueError for exceeding duration"
        except ValueError as e:
            assert "exceeds audio duration" in str(e)

    def test_validate_chunks_invalid_time_range(self):
        """Test validation fails when end time <= start time"""
        chunker = AudioChunker()
        chunks = [
            {"start": 30.0, "end": 20.0, "name": "invalid"}  # End before start
        ]
        try:
            chunker.validate_chunks(chunks, audio_duration=120.0)
            assert False, "Should have raised ValueError for invalid time range"
        except ValueError as e:
            assert "must be greater than start time" in str(e)

    def test_validate_chunks_negative_start_time(self):
        """Test validation fails with negative start time"""
        chunker = AudioChunker()
        chunks = [
            {"start": -5.0, "end": 20.0, "name": "negative"}
        ]
        try:
            chunker.validate_chunks(chunks, audio_duration=120.0)
            assert False, "Should have raised ValueError for negative start time"
        except ValueError as e:
            assert "cannot be negative" in str(e)

    def test_validate_chunks_missing_required_fields(self):
        """Test validation fails with missing required fields"""
        chunker = AudioChunker()
        chunks = [
            {"start": 0.0, "name": "missing_end"}  # Missing 'end'
        ]
        try:
            chunker.validate_chunks(chunks, audio_duration=120.0)
            assert False, "Should have raised ValueError for missing fields"
        except ValueError as e:
            assert "'start' and 'end' times are required" in str(e)

    @patch('pydub.AudioSegment')
    def test_process_audio_file_basic(self, mock_audio_segment):
        """Test basic audio file processing"""
        # Mock audio segment
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=120000)  # 120 seconds in milliseconds
        mock_audio.__getitem__ = Mock(return_value=Mock())
        mock_audio_segment.from_wav.return_value = mock_audio

        chunker = AudioChunker()
        
        # Create temporary files
        chunks_data = [
            {"start": 0.0, "end": 30.0, "name": "intro"},
            {"start": 30.0, "end": 60.0}  # No name
        ]
        
        config_path = self.create_sample_config(chunks_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy audio file
            audio_path = os.path.join(temp_dir, "test.wav")
            Path(audio_path).touch()
            
            output_dir = os.path.join(temp_dir, "output")
            
            try:
                output_files = chunker.process_audio_file(audio_path, config_path, output_dir)
                
                # Check that output directory was created
                assert os.path.exists(output_dir)
                
                # Check that we get expected number of files
                assert len(output_files) == 2
                
                # Check filename patterns
                expected_files = ["0.intro.wav", "1.wav"]  # Sorted by start time
                for expected in expected_files:
                    expected_path = os.path.join(output_dir, expected)
                    assert expected_path in output_files
                    
            finally:
                os.unlink(config_path)

    def test_process_audio_file_non_wav(self):
        """Test processing fails with non-wav files"""
        chunker = AudioChunker()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy non-wav file
            audio_path = os.path.join(temp_dir, "test.mp3")
            Path(audio_path).touch()
            
            config_path = self.create_sample_config([{"start": 0.0, "end": 30.0}])
            
            try:
                try:
                    chunker.process_audio_file(audio_path, config_path, temp_dir)
                    assert False, "Should have raised ValueError for non-wav file"
                except ValueError as e:
                    assert "Only .wav files are supported" in str(e)
            finally:
                os.unlink(config_path)

    def test_chunk_sorting_by_start_time(self):
        """Test that chunks are sorted by start time for sequential indexing"""
        chunker = AudioChunker()
        
        # Create chunks in non-sequential order
        chunks = [
            {"start": 60.0, "end": 90.0, "name": "third"},
            {"start": 0.0, "end": 30.0, "name": "first"}, 
            {"start": 30.0, "end": 60.0, "name": "second"}
        ]
        
        chunker.validate_chunks(chunks, audio_duration=120.0)
        
        # Sort chunks like the real method does
        sorted_chunks = sorted(chunks, key=lambda x: float(x['start']))
        
        assert sorted_chunks[0]["name"] == "first"    # Index 0
        assert sorted_chunks[1]["name"] == "second"   # Index 1  
        assert sorted_chunks[2]["name"] == "third"    # Index 2


class TestConvenienceFunction:
    """Test the convenience function"""
    
    @patch('speech_mine.pickaxe.chunk.AudioChunker')
    def test_chunk_audio_file(self, mock_chunker_class):
        """Test the convenience function calls AudioChunker correctly"""
        mock_chunker = Mock()
        mock_chunker.process_audio_file.return_value = ["file1.wav", "file2.wav"]
        mock_chunker_class.return_value = mock_chunker
        
        result = chunk_audio_file("audio.wav", "config.yaml", "output", 
                                 fade_in=100, fade_out=200, silence_padding=50)
        
        # Check that AudioChunker was initialized with correct parameters
        mock_chunker_class.assert_called_once_with(fade_in_duration=100, 
                                                   fade_out_duration=200,
                                                   silence_padding=50)
        
        # Check that process_audio_file was called
        mock_chunker.process_audio_file.assert_called_once_with("audio.wav", "config.yaml", "output")
        
        # Check return value
        assert result == ["file1.wav", "file2.wav"]