import os
import base64
import torch
import wandb
import json
import time
import numpy as np

from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from diffusers import UNet2DConditionModel
from mosec import Server, Worker, get_logger
from mosec.errors import ClientError
from mosec.mixin import MsgpackMixin

# Setup logging and environment
load_dotenv()
logger = get_logger()

class UNetWorker(MsgpackMixin, Worker):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self._setup_model()

    def _setup_model(self):
        """Initialize model and wandb settings"""
        try:
            self.project = os.getenv("WANDB_PROJECT", "ConditionalUNet")
            self.entity = os.getenv("WANDB_ENTITY")

            # Setup wandb
            wandb.setup(
                settings=wandb.Settings(
                    console="wrap",
                    _disable_stats=True,
                )
            )
            wandb.login(key=os.getenv("WANDB_API_KEY"))

            # Initialize wandb run
            run = wandb.init(
                project=self.project,
                entity=self.entity,
                job_type="inference",
                resume=False,
            )
            if not run:
                raise RuntimeError("Failed to initialize wandb run")

            # Fetch model artifact
            version = os.getenv("MODEL_VERSION", "prod")
            artifact_path = f"unet:{version}"
            if self.entity:
                artifact_path = f"{self.entity}/{self.project}/{artifact_path}"

            logger.info(f"Fetching artifact: {artifact_path}")
            artifact = run.use_artifact(artifact_path, type="model")
            if not artifact:
                raise RuntimeError(f"Failed to fetch artifact: {artifact_path}")

            model_dir = Path(artifact.download())
            logger.info(f"Downloaded model to: {model_dir}")

            # Load model configuration
            config_path = model_dir / "unet_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded model config: {config}")

            # Initialize UNet model
            logger.info("Initializing model...")
            self.model = UNet2DConditionModel(**config)
            weights_path = model_dir / "pytorch_model.bin"

            state_dict = torch.load(
                weights_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self.model.to(device=self.device, dtype=torch.float16)
            self.model.eval()

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Error in setup: {e}", exc_info=True)
            raise

    def _validate_request_fields(self, data: Dict[str, Any]):
        """Validate request data fields"""
        required_fields = ['whisper', 'latent']
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                raise ClientError(f"Missing required field: {field}")
            if not isinstance(data[field], str):
                logger.warning(f"Field {field} must be a base64 string")
                raise ClientError(f"Field {field} must be a base64 string")

    def _decode_base64(self, data: str, field_name: str) -> bytes:
        """Safely decode base64 data"""
        try:
            return base64.b64decode(data)
        except Exception as e:
            logger.warning(f"Invalid base64 encoding for {field_name}: {str(e)}")
            raise ClientError(f"Invalid base64 encoding for {field_name}: {str(e)}")

    def _convert_to_numpy(self, data: bytes, dtype: np.dtype, field_name: str) -> np.ndarray:
        """Convert bytes to numpy array"""
        try:
            return np.frombuffer(data, dtype=dtype)
        except Exception as e:
            logger.warning(f"Could not decode {field_name} as {dtype} array: {str(e)}")
            raise ClientError(f"Could not decode {field_name} as {dtype} array: {str(e)}")

    def _reshape_array(self, array: np.ndarray, shape: tuple, field_name: str) -> np.ndarray:
        """Reshape array with validation"""
        try:
            if array.size != np.prod(shape):
                msg = f"Array size {array.size} does not match expected size {np.prod(shape)}"
                logger.warning(f"{field_name}: {msg}")
                raise ClientError(f"{field_name}: {msg}")
            return array.reshape(shape)
        except Exception as e:
            logger.warning(f"Could not reshape {field_name} to {shape}: {str(e)}")
            raise ClientError(f"Could not reshape {field_name} to {shape}: {str(e)}")

    def _validate_arrays(self, whisper: np.ndarray, latent: np.ndarray):
        """Validate input array shapes and sizes"""
        expected_whisper_shape = (25, 50, 384)
        expected_latent_shape = (25, 8, 32, 32)

        if whisper.shape != expected_whisper_shape:
            msg = f"Invalid whisper dimensions: expected {expected_whisper_shape}, got {whisper.shape}"
            logger.warning(msg)
            raise ClientError(msg)
            
        if latent.shape != expected_latent_shape:
            msg = f"Invalid latent dimensions: expected {expected_latent_shape}, got {latent.shape}"
            logger.warning(msg)
            raise ClientError(msg)

    def forward(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Process a single inference request"""
        try:
            start = time.perf_counter()
            
            try:
                # Validate request fields
                self._validate_request_fields(data)
                
                # Decode base64 input data
                whisper_data = self._decode_base64(data["whisper"], "whisper")
                latent_data = self._decode_base64(data["latent"], "latent")

                # Convert to numpy arrays
                whisper = self._convert_to_numpy(whisper_data, np.float16, "whisper")
                latent = self._convert_to_numpy(latent_data, np.float16, "latent")

                # Reshape arrays
                whisper = self._reshape_array(whisper, (25, 50, 384), "whisper")
                latent = self._reshape_array(latent, (25, 8, 32, 32), "latent")

                # Validate arrays
                self._validate_arrays(whisper, latent)
            except ClientError as e:
                # Mosec will handle ClientError with 400 status
                raise

            # Prepare inputs
            batch_size = whisper.shape[0]
            timesteps = torch.full(
                (batch_size,), 0, device=self.device, dtype=torch.float16
            )
            whisper = torch.from_numpy(whisper).to(device=self.device, dtype=torch.float16)
            latent = torch.from_numpy(latent).to(device=self.device, dtype=torch.float16)

            logger.info(f"Input processing time: {time.perf_counter() - start}s")

            # Run inference
            start = time.perf_counter()
            with torch.no_grad():
                output = self.model(
                    sample=latent,
                    timestep=timesteps,
                    encoder_hidden_states=whisper,
                )
            logger.info(f"Inference time: {time.perf_counter() - start}s")

            # Encode output
            result = output.sample.detach().cpu().numpy().tobytes()
            result_base64 = base64.b64encode(result).decode('utf-8')
            
            return {"output": result_base64}

        except ClientError as e:
            # Client errors (400)
            raise
        except Exception as e:
            # Server errors (500)
            logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise

def main():
    server = Server()
    server.append_worker(
        UNetWorker,
        num=1,
        max_batch_size=1,
        max_wait_time=0,
    )
    server.run()

if __name__ == "__main__":
    main()