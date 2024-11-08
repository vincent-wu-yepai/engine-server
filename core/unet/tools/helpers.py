import wandb

from .log import setup_logger


logger = setup_logger(name="engine-server.tools.helpers")


def cleanup():
    logger.info("Shutting down server and cleaning up resources...")
    wandb.finish()