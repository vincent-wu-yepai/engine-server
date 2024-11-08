import unittest
import base64
import time
import requests
import numpy as np
import msgpack
from typing import Tuple, Dict, Any


class MosecModelTester(unittest.TestCase):
    """Test class for Mosec UNet inference API testing."""
    
    def setUp(self):
        """Setup method to initialize test data."""
        # Load test data
        self.whisper = np.load('data/test_data.npz')
        self.whisper_chunks = np.array(self.whisper["whisper_chunks"][0:25])
        self.latents = np.array(self.whisper["latents"][0:25])
        
        # Convert to float16
        self.whisper_chunks = self.whisper_chunks.astype(np.float16)
        self.latents = self.latents.astype(np.float16)
        
        # Mosec API endpoint (default mosec endpoint)
        self.url = "http://localhost:8000/inference"
        
        # Expected shapes
        self.expected_whisper_shape = (25, 50, 384)
        self.expected_latent_shape = (25, 8, 32, 32)
        self.expected_output_shape = (25, 4, 32, 32)
        
        print(f"Test data loaded - whisper shape: {self.whisper_chunks.shape}, latents shape: {self.latents.shape}")

    def encode_array(self, array: np.ndarray) -> str:
        """Helper method to encode numpy array to base64 string."""
        return base64.b64encode(array.tobytes()).decode('utf-8')

    def decode_response(self, base64_str: str, expected_shape: Tuple[int, ...]) -> np.ndarray:
        """Helper method to decode base64 string to numpy array."""
        output_bytes = base64.b64decode(base64_str)
        array = np.frombuffer(output_bytes, dtype=np.float16)
        return array.reshape(expected_shape)

    def get_error_message(self, response: requests.Response) -> str:
        """Extract error message from response safely."""
        try:
            if response.headers.get('content-type') == 'application/msgpack':
                return str(msgpack.unpackb(response.content))
            return response.content.decode('utf-8')
        except Exception:
            return str(response.content)

    def make_request(self, data: Dict[str, Any]) -> Tuple[requests.Response, float]:
        """Helper method to make API request and measure time."""
        start = time.perf_counter()
        headers = {"Content-Type": "application/msgpack"}
        packed_data = msgpack.packb(data)
        
        try:
            response = requests.post(self.url, data=packed_data, headers=headers)
            if response.status_code == 200:
                # For successful responses, unpack msgpack content
                response._content = msgpack.unpackb(response.content)
            return response, time.perf_counter() - start
        except Exception as e:
            print(f"Request failed: {str(e)}")
            raise

    def test_successful_inference(self):
        """Test successful model inference with valid input."""
        request_data = {
            'whisper': self.encode_array(self.whisper_chunks),
            'latent': self.encode_array(self.latents)
        }

        response, elapsed_time = self.make_request(request_data)
        
        # Assertions
        self.assertEqual(response.status_code, 200, 
                        f"API request failed with status {response.status_code}: {self.get_error_message(response)}")
        
        # Response is now unpacked msgpack
        response_data = response._content
        self.assertIn('output', response_data, "Response missing 'output' field")
        
        # Process response
        response_array = self.decode_response(
            response_data["output"], 
            self.expected_output_shape
        )
        
        # Validate response
        self.assertEqual(
            response_array.shape, 
            self.expected_output_shape, 
            "Incorrect output shape"
        )
        self.assertTrue(np.isfinite(response_array).all(), "Output contains invalid values")
        self.assertFalse(np.all(response_array == 0), "Output contains all zeros")
        
        # Performance validation
        self.assertLess(elapsed_time, 5.0, "Inference took too long")
        print(f"Successful inference time: {elapsed_time:.4f} seconds")

    def test_missing_fields(self):
        """Test error handling for missing required fields."""
        test_cases = [
            ({}, "Missing both fields"),
            ({'whisper': self.encode_array(self.whisper_chunks)}, "Missing latent field"),
            ({'latent': self.encode_array(self.latents)}, "Missing whisper field")
        ]
        
        for data, case_name in test_cases:
            with self.subTest(case=case_name):
                response, _ = self.make_request(data)
                self.assertIn(response.status_code, [400, 422], 
                            f"Expected 400 or 422 for {case_name}, got {response.status_code}: {self.get_error_message(response)}")
                
                error_msg = self.get_error_message(response)
                self.assertTrue(
                    any(err_key in error_msg.lower() for err_key in ['error', 'invalid', 'missing']),
                    f"Error response missing error details: {error_msg}"
                )

    def test_invalid_encoding(self):
        """Test error handling for invalid base64 encoding."""
        test_cases = [
            {'whisper': 'invalid_base64', 'latent': self.encode_array(self.latents)},
            {'whisper': self.encode_array(self.whisper_chunks), 'latent': 'invalid_base64'},
            {'whisper': '', 'latent': ''},
            {'whisper': '!@#$%^&*', 'latent': '!@#$%^&*'}
        ]
        
        for data in test_cases:
            response, _ = self.make_request(data)
            self.assertIn(response.status_code, [400, 422], 
                         f"Expected 400 or 422 for invalid encoding, got {response.status_code}: {self.get_error_message(response)}")

    def test_invalid_shapes(self):
        """Test error handling for arrays with invalid shapes."""
        # Test with wrong batch size
        wrong_batch_whisper = self.whisper_chunks[0:20]  # 20 instead of 25
        wrong_batch_request = {
            'whisper': self.encode_array(wrong_batch_whisper),
            'latent': self.encode_array(self.latents)
        }
        response, _ = self.make_request(wrong_batch_request)
        self.assertIn(response.status_code, [400, 422], 
                     f"Expected 400 or 422 for wrong batch size, got {response.status_code}: {self.get_error_message(response)}")

        # Test with incompatible reshape dimensions
        wrong_shape_whisper = np.random.randn(25, 49, 384).astype(np.float16)
        wrong_shape_request = {
            'whisper': self.encode_array(wrong_shape_whisper),
            'latent': self.encode_array(self.latents)
        }
        response, _ = self.make_request(wrong_shape_request)
        self.assertIn(response.status_code, [400, 422], 
                     f"Expected 400 or 422 for wrong shape, got {response.status_code}: {self.get_error_message(response)}")

    def test_invalid_data_types(self):
        """Test error handling for invalid data types."""
        float32_data = self.whisper_chunks.astype(np.float32)
        wrong_type_request = {
            'whisper': self.encode_array(float32_data),
            'latent': self.encode_array(self.latents)
        }
        response, _ = self.make_request(wrong_type_request)
        self.assertIn(response.status_code, [400, 422], 
                     f"Expected 400 or 422 for wrong data type, got {response.status_code}")

    def test_large_payload(self):
        """Test handling of large payloads."""
        large_whisper = np.tile(self.whisper_chunks, (2, 1, 1))
        large_request = {
            'whisper': self.encode_array(large_whisper),
            'latent': self.encode_array(self.latents)
        }
        response, _ = self.make_request(large_request)
        self.assertIn(response.status_code, [400, 413, 422], 
                     f"Expected 400, 413, or 422 for oversized payload, got {response.status_code}: {self.get_error_message(response)}")

    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        request_data = {
            'whisper': self.encode_array(self.whisper_chunks),
            'latent': self.encode_array(self.latents)
        }
        
        def make_concurrent_request():
            try:
                response, _ = self.make_request(request_data)
                return response
            except Exception as e:
                print(f"Concurrent request failed: {str(e)}")
                raise
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_concurrent_request) for _ in range(5)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        for response in responses:
            self.assertEqual(
                response.status_code, 200, 
                f"Concurrent request failed with status {response.status_code}: {self.get_error_message(response)}"
            )
            response_data = response._content
            self.assertIn('output', response_data, "Response missing 'output' field")


if __name__ == "__main__":
    unittest.main(verbosity=2)