"""Speech-to-text endpoints."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.deps import get_asr_service
from api.schemas.asr import TranscribeResponse
from api.services import ASRService

router = APIRouter(prefix="/v1/asr", tags=["asr"])


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    service: ASRService = Depends(get_asr_service),
) -> TranscribeResponse:
    """Transcribe a voice or audio attachment to plain text."""
    if not service.is_available:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR model is not loaded",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Empty audio payload",
        )

    try:
        text = await service.transcribe(audio_bytes)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return TranscribeResponse(text=text)
