import time
import torch

import numpy as np
from clearml import Model

from zenml.logger import get_logger
from diffusers import UNet2DConditionModel

logger = get_logger(__name__)


class UNetWorker():
    def __init__(self, model_id: str):
        self.device = "cuda"
        self.model_id = model_id
        self._setup_model()

    def _setup_model(self):
        """Initialize UNet model from ClearML artifacts"""
        try:
            logger.info("Loading model from ClearML artifacts...")

            model = Model(model_id=self.model_id)
            model_path = model.get_local_copy()
            
            metadata = model._get_model_data()['design']
            self.model_config = metadata.get('model_config', {})

            self.model = UNet2DConditionModel(**self.model_config)

            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self.model.to(device=self.device, dtype=torch.float16)
            self.model.eval()

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Error in setup: {e}", exc_info=True)
            raise

    def forward(self, whisper: np.ndarray, latent: np.ndarray):
        """Process a single inference request"""
        try:
            start = time.perf_counter()
            
            # Ensure the arrays are on the correct device
            whisper = whisper.to(device=self.device, dtype=torch.float16)
            latent = latent.to(device=self.device, dtype=torch.float16)

            # Ensure array shapes are correct
            if whisper.shape != (25, 50, 384):
                raise ValueError(f"Invalid whisper dimensions: expected (25, 50, 384), got {whisper.shape}")
            if latent.shape != (25, 8, 32, 32):
                raise ValueError(f"Invalid latent dimensions: expected (25, 8, 32, 32), got {latent.shape}")

            # Prepare inputs
            timesteps = torch.tensor([0], device=self.device, dtype=self.dtype)

            # Run inference
            start = time.perf_counter()
            with torch.no_grad():
                output = self.model(
                    sample=latent,
                    timestep=timesteps,
                    encoder_hidden_states=whisper,
                ).sample
            
            logger.info(f"Inference time: {time.perf_counter() - start}s")

            # Pass the tensor directly to next worker
            return output
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise