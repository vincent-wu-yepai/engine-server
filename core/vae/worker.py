import torch
import time

import numpy as np

from diffusers import AutoencoderTiny
from mosec import SSEWorker, get_logger
from mosec.mixin import RedisShmIPCMixin


logger = get_logger()


class VAEWorker(RedisShmIPCMixin, SSEWorker):
    """VAE worker"""

    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self._setup_model()

    def _setup_model(self):
        """Initialize VAE model"""
        try:
            logger.info("Initializing VAE model...")
            self.model = AutoencoderTiny.from_pretrained("madebyollin/taesd")
            self.model.encoder = None
            self.model.to(device=self.device)
            self.model.eval()
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
            latents = (1 / self.model.config.scaling_factor) * latents
            
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