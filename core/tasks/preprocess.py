import numpy as np

from prefect import task


@task
def transform_audio_bytes_to_np_array(audio: bytes) -> np.ndarray:
    audio = np.frombuffer(audio, dtype=np.float32)
    return audio
