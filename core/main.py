import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.routers import router as inference_router
from core.workers.avatar import AvatarWorker

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up the application...")
    
    yield
    
    print("Shutting down the application...")


app = FastAPI(lifespan=lifespan)

app.include_router(inference_router, prefix="/avatar", tags=["inference"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
