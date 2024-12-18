from fastapi import APIRouter, WebSocket

from core.flows.inference import inference_flow

router = APIRouter()


@router.websocket("/inference")
async def websocket_inference_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            avatar_id: str = await websocket.receive_text()
            audio_data: bytes = await websocket.receive_bytes()
            await inference_flow(avatar_id, audio_data)
    except Exception as e:
        print(f"WebSocket错误: {str(e)}")
    finally:
        await websocket.close()