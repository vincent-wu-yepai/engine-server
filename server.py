import subprocess
import signal
import logging
import wandb

from mosec import Server
from mosec.mixin import RedisShmIPCMixin

from core.unet.worker import UNetWorker
from core.vae.worker import VAEWorker
from core.sse.worker import SSEResponseWorker


logger = logging.getLogger(__name__)


def cleanup():
    logger.info("Shutting down server and cleaning up resources...")
    wandb.finish()


def main():
    # Register cleanup on signals
    signal.signal(signal.SIGINT, lambda s, f: cleanup())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup())
    
    with subprocess.Popen([
        "redis-server",
        "--save", "",  # Disable RDB
        "--appendonly", "no"  # Disable AOF
    ]) as p:
        # configure the redis url
        RedisShmIPCMixin.set_redis_url("redis://localhost:6379/0")
        server = Server()
        server.register_daemon("redis-server", p)
        
        # Add UNet worker
        server.append_worker(
            UNetWorker,
            num=1,
            max_batch_size=1,
            max_wait_time=0,
            timeout=10
        )
        
        # Add VAE worker
        server.append_worker(
            VAEWorker,
            num=1,
            max_batch_size=1,
            max_wait_time=0,
            timeout=10
        )
        
        # Add SSE worker
        server.append_worker(
            SSEResponseWorker,
            num=1,
            max_batch_size=1,
            max_wait_time=0,
            timeout=10  
        )
        
        server.run()


if __name__ == "__main__":
    main()