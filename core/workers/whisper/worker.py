import os
import time
import torch
import tempfile

import numpy as np
from clearml import Model

from zenml.logger import get_logger
from .model import load_model

logger = get_logger(__name__)


class WhisperFeatureWorker():
    def __init__(self, model_id: str):
        self.device = "cuda"
        self.model_id = model_id
        self._setup_model()
        
    def _setup_model(self):
        """Initialize Whisper model from ClearML artifacts"""
        try:
            logger.info("Loading model from ClearML artifacts...")

            model = Model(model_id=self.model_id)
            model_path = model.get_local_copy()
            
            metadata = model._get_model_data()['design']
            self.model_type = metadata.get('model_type', 'tiny')
            
            self.model = load_model(model_path, device=torch.device(self.device))
            logger.info(f"Model ({self.model_type}) loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def get_sliced_feature(self, feature_array: np.ndarray, vid_idx: int, 
                          audio_feat_length: list = [2, 2], fps: int = 25) -> tuple:
        """Get sliced features based on a given index"""
        try:
            length = len(feature_array)
            selected_feature = []
            selected_idx = []
            
            center_idx = int(vid_idx * 50/fps)
            left_idx = center_idx - audio_feat_length[0] * 2
            right_idx = center_idx + (audio_feat_length[1] + 1) * 2
            
            for idx in range(left_idx, right_idx):
                idx = max(0, idx)
                idx = min(length-1, idx)
                x = feature_array[idx]
                selected_feature.append(x)
                selected_idx.append(idx)
            
            selected_feature = np.concatenate(selected_feature, axis=0)
            selected_feature = selected_feature.reshape(-1, 384)  # 50*384
            return selected_feature, selected_idx
            
        except Exception as e:
            logger.error(f"Error in feature slicing: {e}", exc_info=True)
            raise

    def feature2chunks(self, feature_array: np.ndarray, fps: int, 
                      audio_feat_length: list = [2, 2]) -> list:
        """Convert feature array to chunks"""
        try:
            whisper_chunks = []
            whisper_idx_multiplier = 50./fps
            i = 0
            logger.info(f"Processing video in {fps} FPS, audio idx in 50FPS")
            
            while True:
                start_idx = int(i * whisper_idx_multiplier)
                if start_idx >= len(feature_array):
                    break
                selected_feature, _ = self.get_sliced_feature(
                    feature_array=feature_array,
                    vid_idx=i,
                    audio_feat_length=audio_feat_length,
                    fps=fps
                )
                whisper_chunks.append(selected_feature)
                i += 1

            return whisper_chunks
            
        except Exception as e:
            logger.error(f"Error in chunk conversion: {e}", exc_info=True)
            raise

    def forward(self, audio_binary_data: bytes, fps: int = 25) -> list:
        """Process audio data and extract features"""
        try:
            start = time.perf_counter()
            logger.info("Processing audio data...")

            # Create temporary file for audio data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
                tmpfile.write(audio_binary_data)
                temp_audio_path = tmpfile.name

            try:
                # Extract features using Whisper
                whisper_feature = self.audio2feat(temp_audio_path)
                
                # Convert features to chunks
                whisper_chunks = self.feature2chunks(
                    feature_array=whisper_feature,
                    fps=fps
                )
                
                logger.info(f"Feature extraction time: {time.perf_counter() - start:.3f}s")
                return whisper_chunks

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_audio_path)
                except Exception as e:
                    logger.warning(f"Error removing temporary file: {e}")

        except Exception as e:
            logger.error(f"Error in Whisper forward pass: {str(e)}", exc_info=True)
            raise

    def audio2feat(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file"""
        try:
            result = self.model.transcribe(audio_path)
            embed_list = []

            for emb in result['segments']:
                encoder_embeddings = emb['encoder_embeddings']
                encoder_embeddings = encoder_embeddings.transpose(0, 2, 1, 3)
                encoder_embeddings = encoder_embeddings.squeeze(0)
                
                start_idx = int(emb['start'])
                end_idx = int(emb['end'])
                emb_end_idx = int((end_idx - start_idx)/2)
                embed_list.append(encoder_embeddings[:emb_end_idx])

            concatenated_array = np.concatenate(embed_list, axis=0)
            return concatenated_array
            
        except Exception as e:
            logger.error(f"Error in audio feature extraction: {e}", exc_info=True)
            raise