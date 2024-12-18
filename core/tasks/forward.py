import numpy as np

from zenml.steps import step


@step
def forward(audio: np.ndarray, latent: np.ndarray) -> np.ndarray:
    pass
