import base64
import json

import numpy as np

from mosec import SSEWorker, get_logger
from mosec.mixin import RedisShmIPCMixin


logger = get_logger()


class SSEResponseWorker(RedisShmIPCMixin, SSEWorker):
    """SSE streaming support"""
    resp_mime_type = "text/event-stream"

    def __init__(self):
        super().__init__()

    def forward(self, images: np.ndarray):
        """Process VAE decoding and stream results"""
        # Stream each image
        length = len(images)
        for i, image in enumerate(images):
            # Convert image to base64 to make it more efficient to transmit
            img_bytes = base64.b64encode(image.tobytes()).decode('utf-8')
                
            # Create event data with shape information
            event_data = {
                'index': i,
                'shape': image.shape,
                'dtype': str(image.dtype),
                'data': img_bytes
            }
                
            # Send event
            self.send_stream_event(
                text=json.dumps(event_data),
                index=0
            )
            logger.info(f"Processed image {i} / {length}")
            
        logger.info("All images streamed successfully")
        return None