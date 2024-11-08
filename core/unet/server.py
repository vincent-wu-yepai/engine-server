import os
import base64
import torch
import wandb
import json
import time
import atexit

import numpy as np
import litserve as ls

from pathlib import Path
from dotenv import load_dotenv
from diffusers import UNet2DConditionModel
from litserve.server import HTTPException

from tools.log import setup_logger
from tools.helpers import cleanup


load_dotenv()
logger = setup_logger(name="engine-server.core.unet")
atexit.register(cleanup)


class UNetLitAPI(ls.LitAPI):
    def setup(self, device):
        """Initialization setup - called once at startup"""
        try:
            self.device = device
            self.project = os.getenv("WANDB_PROJECT", "ConditionalUNet")
            self.entity = os.getenv("WANDB_ENTITY")

            wandb.setup(
                settings=wandb.Settings(
                    console="wrap",  # Ensure logs wrap correctly
                    _disable_stats=True,  # Disable statistics collection
                )
            )

            # Log in to wandb
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

            # Fetch the model
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

            # Load configuration
            config_path = model_dir / "unet_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded model config: {config}")

            # Initialize model
            logger.info("Initializing model...")
            self.model = UNet2DConditionModel(**config)
            weights_path = model_dir / "pytorch_model.bin"

            state_dict = torch.load(
                weights_path, map_location=device, weights_only=True
            )

            self.model.load_state_dict(state_dict)
            self.model.to(device=device, dtype=torch.float16)
            self.model.eval()  # Set to evaluation mode

            logger.info("Model initialized successfully")

        except Exception as e:
            logger.error(f"Error in setup: {e}", exc_info=True)
            raise

    def decode_request(self, request):
        """Decode request data"""
        try:
            # verify request data
            start = time.perf_counter()
            if "whisper" not in request or "latent" not in request:
                raise HTTPException(
                    status_code=400,
                    detail="Missing required fields: 'whisper' and 'latent' are required"
                )

            try:
                whisper_data = base64.b64decode(request["whisper"])
                latent_data = base64.b64decode(request["latent"])
            except:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid base64 encoding in request data"
                )
            logger.info(f"Request decoding time: {time.perf_counter() - start}s")
            start = time.perf_counter()
            try:
                whisper = np.frombuffer(whisper_data, dtype=np.float16)
                latent = np.frombuffer(latent_data, dtype=np.float16)
            except:
                raise HTTPException(
                    status_code=400,
                    detail="Could not decode data as float16 arrays"
                )

            # Calculate expected sizes
            expected_whisper_size = 25 * 50 * 384
            expected_latent_size = 25 * 8 * 32 * 32

            # Validate array sizes before reshape
            if whisper.size != expected_whisper_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid whisper array size: expected {expected_whisper_size}, got {whisper.size}"
                )
            
            if latent.size != expected_latent_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid latent array size: expected {expected_latent_size}, got {latent.size}"
                )
            
            try:
                whisper = whisper.reshape(25, 50, 384)
                latent = latent.reshape(25, 8, 32, 32)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not reshape arrays to expected dimensions: {str(e)}"
                )

            # Additional shape validations
            if len(whisper.shape) != 3:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid whisper shape: expected (25, 50, 384), got {whisper.shape}"
                )
            
            if len(latent.shape) != 4:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid latent shape: expected (25, 8, 32, 32), got {latent.shape}"
                )
            
            if whisper.shape != (25, 50, 384):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid whisper dimensions: expected (25, 50, 384), got {whisper.shape}"
                )
                
            if latent.shape != (25, 8, 32, 32):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid latent dimensions: expected (25, 8, 32, 32), got {latent.shape}"
                )

            # Prepare time steps
            batch_size = whisper.shape[0]
            timesteps = torch.full(
                (batch_size,), 0, device=self.device, dtype=torch.float16
            )
            whisper = torch.from_numpy(whisper).to(device=self.device, dtype=torch.float16)
            latent = torch.from_numpy(latent).to(device=self.device, dtype=torch.float16)
            logger.info(f"Request validation time: {time.perf_counter() - start}s")
            return {
                "sample": latent,
                "timestep": timesteps,
                "encoder_hidden_states": whisper,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"engine-server.core.unet.server.decode-request.error: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error during request processing"
            )

    def predict(self, inputs):
        """Execute prediction"""
        try:
            with torch.no_grad():
                start = time.perf_counter()
                output = self.model(
                    sample=inputs["sample"],
                    timestep=inputs["timestep"],
                    encoder_hidden_states=inputs["encoder_hidden_states"],
                )
                logger.info(f"Inference time: {time.perf_counter() - start}s")
            return output.sample.to(dtype=torch.float16)

        except Exception as e:
            logger.error(f"engine-server.core.unet.server.predict.error: {e}", exc_info=True)
            raise

    def encode_response(self, output):
        """Encode response data"""
        try:
            result = output.detach().cpu().numpy().tobytes()
            result_base64 = base64.b64encode(result).decode('utf-8')
            return {"output": result_base64}
        except Exception as e:
            logger.error(f"Error in encode_response: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    # Configure server
    server = ls.LitServer(
        UNetLitAPI(),
        accelerator="auto",  # Automatically select available accelerator
        max_batch_size=1,  # Set maximum batch size
        workers_per_device=1
    )

    # Start Server
    port = int(os.getenv("PORT", 8000))
    server.run(port=port)
