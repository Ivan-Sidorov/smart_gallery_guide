"""Exhibit search/recognize endpoints."""

import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)

from api.deps import get_exhibit_service
from api.schemas.exhibits import (
    ExhibitDTO,
    ExhibitSearchRequest,
    ExhibitSearchResultDTO,
)
from api.services import ExhibitService

router = APIRouter(prefix="/v1/exhibits", tags=["exhibits"])


@router.get("", response_model=list[ExhibitDTO])
async def list_exhibits(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: ExhibitService = Depends(get_exhibit_service),
) -> list[ExhibitDTO]:
    """List exhibits ordered by created_at desc."""
    return await service.list(limit=limit, offset=offset)


@router.get("/{exhibit_id}", response_model=ExhibitDTO)
async def get_exhibit(
    exhibit_id: str,
    service: ExhibitService = Depends(get_exhibit_service),
) -> ExhibitDTO:
    """Fetch a single exhibit by id."""
    dto = await service.get(exhibit_id)
    if dto is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Exhibit not found")
    return dto


@router.post("/search", response_model=list[ExhibitSearchResultDTO])
async def search_exhibits(
    payload: ExhibitSearchRequest,
    service: ExhibitService = Depends(get_exhibit_service),
) -> list[ExhibitSearchResultDTO]:
    """Cascading text search (title –> description) over ChromaDB + BM25."""
    return await service.search_by_text(
        query=payload.query,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
    )


@router.post("/recognize", response_model=list[ExhibitSearchResultDTO])
async def recognize_exhibit(
    image: UploadFile = File(...),
    top_k: int | None = Form(default=None),
    score_threshold: float | None = Form(default=None),
    user_id: int | None = Form(default=None),
    session_id: uuid.UUID | None = Form(default=None),
    service: ExhibitService = Depends(get_exhibit_service),
) -> list[ExhibitSearchResultDTO]:
    """Recognise an exhibit from a user photo (SigLIP + Chroma)."""
    image_bytes = await image.read()
    return await service.recognize_by_image(
        image_bytes=image_bytes,
        top_k=top_k,
        score_threshold=score_threshold,
    )
