"""FAQ search endpoint."""

from fastapi import APIRouter, Depends

from api.deps import get_faq_service
from api.schemas.faq import FAQSearchRequest, FAQSearchResultDTO
from api.services import FAQService

router = APIRouter(prefix="/v1/faq", tags=["faq"])


@router.post("/search", response_model=list[FAQSearchResultDTO])
async def search_faq(
    payload: FAQSearchRequest,
    service: FAQService = Depends(get_faq_service),
) -> list[FAQSearchResultDTO]:
    """Search FAQ items inside a single exhibit by question text."""
    return await service.search(
        exhibit_id=payload.exhibit_id,
        question=payload.question,
        top_k=payload.top_k,
        score_threshold=payload.score_threshold,
    )
