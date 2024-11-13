import unittest
import base64
import time
import requests
import numpy as np
import json
from typing import Tuple, Dict, Any, List, Optional
import logging
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MosecPipelineTester(unittest.TestCase):
    """Test class for Mosec UNet+VAE inference pipeline testing."""
    
    def setUp(self):
        """Setup method to initialize test data."""
        # Load test data
        self.whisper = np.load('data/test_data.npz')
        self.whisper_chunks = np.array(self.whisper["whisper_chunks"][0:25])
        self.latents = np.array(self.whisper["latents"][0:25])
        
        # Convert to float16
        self.whisper_chunks = self.whisper_chunks.astype(np.float16)
        self.latents = self.latents.astype(np.float16)
        
        # Mosec API endpoint
        self.url = "http://localhost:8000/inference"
        
        # Expected shapes
        self.expected_whisper_shape = (25, 50, 384)
        self.expected_latent_shape = (25, 8, 32, 32)
        self.expected_image_shape = (256, 256, 3)  # VAE outputs 256x256 images
        
        logger.info(f"Test data loaded - whisper shape: {self.whisper_chunks.shape}, latents shape: {self.latents.shape}")

    def encode_array(self, array: np.ndarray) -> str:
        """Helper method to encode numpy array to base64 string."""
        return base64.b64encode(array.tobytes()).decode('utf-8')

    def process_sse_event(self, line: bytes) -> Optional[Dict[str, Any]]:
        """Process a single SSE event line."""
        if not line or not line.strip():
            return None
            
        # Remove 'data: ' prefix if present
        if line.startswith(b'data: '):
            line = line[6:]
        
        try:
            event_data = json.loads(line)
            return event_data
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse SSE event: {e}")
            return None

    def make_request(self) -> Tuple[bool, float, float, List[np.ndarray]]:
        """Helper method to make API request and measure first response latency and FPS."""
        start_time = time.perf_counter()
        first_response_latency = None
        last_response_time = None
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        request_data = {
            'whisper': self.encode_array(self.whisper_chunks),
            'latent': self.encode_array(self.latents)
        }
        
        try:
            response = requests.post(
                self.url,
                json=request_data,
                headers=headers,
                stream=True,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.error(f"Request failed with status {response.status_code}")
                logger.error(f"Response content: {response.content}")
                return False, -1, -1, []
            
            images = []
            image_dict = {}
            
            for line in response.iter_lines():
                event_data = self.process_sse_event(line)
                if not event_data:
                    continue
                    
                current_time = time.perf_counter()
                last_response_time = current_time  # Update last response time
                
                try:
                    # Extract image information
                    index = event_data['index']
                    shape = tuple(event_data['shape'])
                    dtype = np.dtype(event_data['dtype'])
                    
                    if first_response_latency is None:
                        first_response_latency = current_time - start_time
                        logger.info(f"First response latency: {first_response_latency:.4f}s")
                    
                    img_bytes = base64.b64decode(event_data['data'])
                    image = np.frombuffer(img_bytes, dtype=dtype).reshape(shape)
                    image_dict[index] = image
                    
                except Exception as e:
                    logger.error(f"Error processing image data: {e}")
                    continue

            images = [image_dict[i] for i in sorted(image_dict.keys())]
            
            # Calculate FPS
            if last_response_time and first_response_latency is not None:
                processing_time = last_response_time - (start_time + first_response_latency)
                fps = len(images) / processing_time if processing_time > 0 else 0
            else:
                fps = 0
            
            return True, first_response_latency or -1, fps, images
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return False, -1, -1, []

    def validate_image(self, image: np.ndarray):
        """Validate a single image."""
        self.assertEqual(image.shape, self.expected_image_shape, 
                        f"Invalid image shape: {image.shape}")
        self.assertEqual(image.dtype, np.dtype('uint8'), 
                        f"Invalid image dtype: {image.dtype}")
        self.assertTrue(np.all(image >= 0) and np.all(image <= 255), 
                        "Image values out of valid range")

    def test_successful_inference(self):
        """Test successful model inference with valid input."""
        success, first_response_latency, fps, images = self.make_request()
        
        self.assertTrue(success, "Request failed")
        self.assertGreater(first_response_latency, 0, "First response latency not recorded")
        self.assertLess(first_response_latency, 5.0, 
                       f"First response latency too high: {first_response_latency:.4f}s")
        
        self.assertEqual(len(images), 25, f"Incorrect number of images returned: {len(images)}")
        
        logger.info(f"Achieved FPS: {fps:.2f}")
        
        for idx, image in enumerate(images):
            with self.subTest(image_index=idx):
                self.validate_image(image)

    @classmethod
    def make_concurrent_request(cls):
        """Static method for process pool execution."""
        tester = cls()
        tester.setUp()
        try:
            return tester.make_request()
        except Exception as e:
            logger.error(f"Concurrent request failed: {str(e)}")
            raise

    def test_concurrent_requests(self):
        """Test handling of concurrent requests with first response latency and FPS tracking using process pool."""
        from concurrent.futures import ProcessPoolExecutor
        
        first_response_latencies = []
        fps_results = []
        num_requests = 15
        
        # Use ProcessPoolExecutor instead of ThreadPoolExecutor
        with ProcessPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(self.make_concurrent_request) for _ in range(num_requests)]
            results = [future.result() for future in futures]
        
        # Process results
        for idx, (success, latency, fps, images) in enumerate(results):
            self.assertTrue(success, "Request failed")
            self.assertEqual(len(images), 25, "Incorrect number of images returned")
            first_response_latencies.append(latency)
            fps_results.append(fps)
            
            for image in images:
                self.validate_image(image)
        
        # Calculate and log statistics
        latencies = np.array(first_response_latencies)
        fps_array = np.array(fps_results)
        
        logger.info("\nPerformance statistics across concurrent requests:")
        logger.info("\nFirst response latency statistics:")
        logger.info(f"  mean: {np.mean(latencies):.4f}s")
        logger.info(f"  std: {np.std(latencies):.4f}s")
        logger.info(f"  min: {np.min(latencies):.4f}s")
        logger.info(f"  max: {np.max(latencies):.4f}s")
        logger.info(f"  p50: {np.percentile(latencies, 50):.4f}s")
        logger.info(f"  p90: {np.percentile(latencies, 90):.4f}s")
        logger.info(f"  p99: {np.percentile(latencies, 99):.4f}s")
        
        logger.info("\nFPS statistics:")
        logger.info(f"  mean: {np.mean(fps_array):.2f}")
        logger.info(f"  std: {np.std(fps_array):.2f}")
        logger.info(f"  min: {np.min(fps_array):.2f}")
        logger.info(f"  max: {np.max(fps_array):.2f}")
        logger.info(f"  p50: {np.percentile(fps_array, 50):.2f}")
        logger.info(f"  p90: {np.percentile(fps_array, 90):.2f}")
        logger.info(f"  p99: {np.percentile(fps_array, 99):.2f}")
        
        # Log individual request FPS
        logger.info("\nPer-request FPS:")
        for idx, fps in enumerate(fps_results):
            logger.info(f"  Request {idx + 1}: {fps:.2f} FPS")
            if fps < 25:
                logger.warning(f"  Request {idx + 1} did not achieve target 25 FPS")


if __name__ == "__main__":
    # Set start method for ProcessPoolExecutor
    mp.set_start_method('spawn')
    unittest.main(verbosity=2)