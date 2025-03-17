from fastapi import APIRouter

router = APIRouter(prefix="/api")


@router.get("/healthz")
async def health_check():
    """Health check endpoint.

    Returns:
        dict: Status information
    """
    return {"status": "healthy"}
