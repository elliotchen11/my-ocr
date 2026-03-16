from fastapi import APIRouter

router = APIRouter(prefix="/api/models", tags=["models"])

MODELS = [
    {
        "modelname": "mistral-small3.2",
        "description": "Vision-capable model used for OCR processing of images and PDF pages",
    },
    {
        "modelname": "gpt-oss",
        "description": "Text model used for structured field extraction from OCR output",
    },
]


@router.get("")
async def get_models():
    return MODELS
