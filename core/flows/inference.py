from prefect import flow

from core.tasks.preprocess import transform_audio_bytes_to_np_array


@flow
async def inference_flow(avatar_id: str, audio_data: bytes):
    audio_np_array = transform_audio_bytes_to_np_array(audio_data)
    pass
