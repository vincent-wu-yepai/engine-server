from prefect import task
from prefect.logging import get_run_logger



class AvatarLoaderWorker():

    @task
    def __init__(self, avatar_ids: list[str]):
        self.avatar_ids = avatar_ids
        self.avatars = self._setup_avatar()
        self.logger = get_run_logger()

    @task
    def _setup_model(self):
        """Download all avatar files from S3"""
        try:
            self.logger.info("Loading avatars from S3...")
            for avatar_id in self.avatar_ids:
                avatar = Avatar(avatar_id)
                self.avatars.append(avatar)

            self.logger.info("Model initialized successfully")

        except Exception as e:
            self.logger.error(f"Error in setup: {e}", exc_info=True)
            raise

    @task
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