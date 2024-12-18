import time
import torch
import numpy as np

from clearml import Model
from zenml.logger import get_logger
from diffusers import AutoencoderTiny

logger = get_logger(__name__)


class VAEWorker():
    """VAE worker"""

    def __init__(self, model_id: str):
        self.device = "cuda"
        self.model_id = model_id
        self._setup_model()

    def _setup_model(self):
        """Initialize VAE model"""
        try:
            logger.info("Loading VAE model from ClearML artifacts...")

            model = Model(model_id=self.model_id)
            model_path = model.get_local_copy()

            metadata = model._get_model_data()['design']
            self.model_config = metadata.get('model_config', {})
            self.scaling_factor = metadata.get('scaling_factor', 1.0)

            self.model = AutoencoderTiny(**self.model_config)

            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self.model.to(device=self.device, dtype=torch.float16)
            self.model.eval()
            self.model.encoder = None
            logger.info("VAE model initialized successfully")
        except Exception as e:
            logger.error(f"Error in VAE setup: {e}", exc_info=True)
            raise

    def forward(self, latents: torch.Tensor) -> np.ndarray:
        """Process VAE decoding and stream results"""
        try:
            start = time.perf_counter()
            batch_size = latents.shape[0]
            logger.info(f"Processing batch of {batch_size} images")
            
            # Scale latents according to VAE configuration
            latents = latents / self.scaling_factor
            
            # Decode latents to image
            with torch.no_grad():
                images = self.model.decode(latents.to(self.model.dtype)).sample
            images = (images / 2 + 0.5).clamp(0, 1)
            images = images.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            images = (images * 255).round().astype("uint8")
            images = images[..., ::-1]  # RGB to BGR
            logger.info(f"VAE decode time: {time.perf_counter() - start:.3f}s")
            return images

        except Exception as e:
            logger.error(f"Error in VAE forward pass: {str(e)}", exc_info=True)
            raise