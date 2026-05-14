import numpy as np
from fastapi import APIRouter, File, Form, UploadFile, HTTPException,Request
from typing import Optional
from src.models.clip_embedder import CLIPEmbedder
from src.models.text_embedder import CLIPTextEmbedder
from src.retrieval.query_parser import parse_filters
from src.retrieval.search_engine import search
from src.core.limiter import limiter
import os
import google.generativeai as genai
from pydantic import BaseModel

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')
class ChatRequest(BaseModel):
    message: str
    context: str


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

    # --- Vector Search Logic ---
    if image and query:
        img_vec = image_embedder.embed_image_bytes(await image.read())
        txt_vec = text_embedder.embed_text(query)
        combined = np.array(img_vec) + np.array(txt_vec)
        combined /= np.linalg.norm(combined)  
        query_vector = combined.tolist()
    elif image:
        query_vector = image_embedder.embed_image_bytes(await image.read())
        query = "Visually similar products" # Default text for Gemini if only image used
    else:
        query_vector = text_embedder.embed_text(query)

    parsed_filters = await parse_filters(query) if query else {}
    results = search(query_vector, parsed_filters, top_k=top_k)

    
    product_context = ""
    for item in results[:5]: # Only send top 5 to Gemini to keep it fast and cheap
        # Handle both dict and object types safely based on Qdrant return style
        payload = item.get("payload", {}) if isinstance(item, dict) else getattr(item, "payload", {})
        title = payload.get("product_name", "Unknown Product")
        price = payload.get("price", "N/A")
        product_context += f"- {title} (Price: ${price})\n"

    prompt = f"""
    You are a helpful, conversational e-commerce sales assistant.
    The user searched for: '{query}'
    Here are the top products our database retrieved:
    {product_context}
    
    Write a short, friendly 2-3 sentence response recommending these specific options. 
    Compare them briefly to help the user choose. Do not make up any products or prices.
    Keep it concise.
    """

    try:
        ai_response = model.generate_content(prompt)
        ai_message = ai_response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        ai_message = "I found some great options for you below!" # Fallback

    
    return {
        "results": results, 
        "filters_applied": parsed_filters,
        "ai_message": ai_message
    }

@router.post("/chat")
async def chat_with_ai(request: ChatRequest):
    prompt = f"""
    You are a helpful e-commerce assistant. The user is looking at these products:
    {request.context}
    
    The user says: "{request.message}"
    
    Respond directly, concisely, and helpfully. Do not use complex formatting.
    """
    try:
        ai_response = model.generate_content(prompt)
        return {"reply": ai_response.text}
    except Exception as e:
        print(f"Chat Error: {e}")
        return {"reply": "Sorry, I am having trouble connecting to my servers right now."}