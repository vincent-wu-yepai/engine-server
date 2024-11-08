import unittest
import base64
import time
import requests
import numpy as np
from typing import Tuple, Dict, Any


class ModelTester(unittest.TestCase):
    """Test class for model inference API testing."""
    
    def setUp(self):
        """Setup method to initialize test data."""
        # Load test data
        self.whisper = np.load('data/test_data.npz')
        self.whisper_chunks = np.array(self.whisper["whisper_chunks"][0:25])
        self.latents = np.array(self.whisper["latents"][0:25])
        
        # Convert to float16
        self.whisper_chunks = self.whisper_chunks.astype(np.float16)
        self.latents = self.latents.astype(np.float16)
        
        # API endpoint
        self.url = "http://localhost:8000/predict"
        
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

    def make_request(self, data: Dict[str, Any]) -> Tuple[requests.Response, float]:
        """Helper method to make API request and measure time."""
        start = time.perf_counter()
        response = requests.post(self.url, json=data)
        elapsed = time.perf_counter() - start
        return response, elapsed

    def test_successful_inference(self):
        """Test successful model inference with valid input."""
        request_data = {
            'whisper': self.encode_array(self.whisper_chunks),
            'latent': self.encode_array(self.latents)
        }

        response, elapsed_time = self.make_request(request_data)
        
        # Assertions
        self.assertEqual(response.status_code, 200, "API request failed")
        self.assertIn('output', response.json(), "Response missing 'output' field")
        
        # Process response
        response_array = self.decode_response(
            response.json()["output"], 
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
                self.assertEqual(response.status_code, 400, f"Expected 400 for {case_name}")
                self.assertIn('detail', response.json(), "Error response missing detail")

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
            self.assertEqual(response.status_code, 400, "Expected 400 for invalid encoding")

    def test_invalid_shapes(self):
        """Test error handling for arrays with invalid shapes."""
        # Test with wrong batch size
        wrong_batch_whisper = self.whisper_chunks[0:20]  # 20 instead of 25
        wrong_batch_request = {
            'whisper': self.encode_array(wrong_batch_whisper),
            'latent': self.encode_array(self.latents)
        }
        response, _ = self.make_request(wrong_batch_request)
        self.assertEqual(response.status_code, 400, "Expected 400 for wrong batch size")

        # Test with incompatible reshape dimensions
        wrong_shape_whisper = np.random.randn(25, 49, 384).astype(np.float16)  # Wrong middle dimension
        wrong_shape_request = {
            'whisper': self.encode_array(wrong_shape_whisper),
            'latent': self.encode_array(self.latents)
        }
        response, _ = self.make_request(wrong_shape_request)
        self.assertEqual(response.status_code, 400, "Expected 400 for incompatible dimensions")

    def test_invalid_data_types(self):
        """Test error handling for invalid data types."""
        # Test with different data types
        float32_data = self.whisper_chunks.astype(np.float32)
        wrong_type_request = {
            'whisper': self.encode_array(float32_data),
            'latent': self.encode_array(self.latents)
        }
        response, _ = self.make_request(wrong_type_request)
        self.assertNotEqual(response.status_code, 200, "Expected error for wrong data type")

    def test_large_payload(self):
        """Test handling of large payloads."""
        # Create a large array (2x normal size)
        large_whisper = np.tile(self.whisper_chunks, (2, 1, 1))
        large_request = {
            'whisper': self.encode_array(large_whisper),
            'latent': self.encode_array(self.latents)
        }
        response, _ = self.make_request(large_request)
        self.assertEqual(response.status_code, 400, "Expected 400 for oversized payload")

    def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        import concurrent.futures
        
        request_data = {
            'whisper': self.encode_array(self.whisper_chunks),
            'latent': self.encode_array(self.latents)
        }
        
        def make_concurrent_request():
            return requests.post(self.url, json=request_data)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_concurrent_request) for _ in range(5)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # Verify all responses
        for response in responses:
            self.assertEqual(response.status_code, 200, "Concurrent request failed")


def run_tests():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()