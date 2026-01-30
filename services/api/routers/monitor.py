from fastapi import APIRouter
router = APIRouter()

@router.get("")
def monitor():
    # TODO: run drift checks
    return {"drift": False, "note": "placeholder"}
