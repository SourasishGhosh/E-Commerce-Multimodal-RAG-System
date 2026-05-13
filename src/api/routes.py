import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException,Request
from typing import Optional
from src.models.clip_embedder import CLIPEmbedder
from src.models.text_embedder import CLIPTextEmbedder
from src.retrieval.query_parser import parse_filters
from src.retrieval.search_engine import search
from src.core.limiter import limiter

router = APIRouter()
image_embedder = CLIPEmbedder()
text_embedder = CLIPTextEmbedder()

@router.post("/search")
@limiter.limit("5/minute")
async def search_products(
    request: Request,
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(10)
):
    if not query and not image:
        raise HTTPException(status_code=400, detail="Provide text query, image, or both.")

    
    if image and query:
        img_vec = image_embedder.embed_image_bytes(await image.read())
        txt_vec = text_embedder.embed_text(query)
        combined = np.array(img_vec) + np.array(txt_vec)
        combined /= np.linalg.norm(combined)  
        query_vector = combined.tolist()

    elif image:
        query_vector = image_embedder.embed_image_bytes(await image.read())
        query = ""

    else:
        query_vector = text_embedder.embed_text(query)

    
    parsed_filters = await parse_filters(query) if query else {}

    results = search(query_vector, parsed_filters, top_k=top_k)
    return {"results": results, "filters_applied": parsed_filters}