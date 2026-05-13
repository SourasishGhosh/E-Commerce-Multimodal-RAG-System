import json
import logging
import google.generativeai as genai
from src.core.config import settings

logger = logging.getLogger(__name__)

genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

SYSTEM_PROMPT = """
You extract filter criteria from a user's product search query.
Return ONLY a valid JSON object with these keys (omit keys that aren't mentioned):
  price_max (number), price_min (number), color (string), style (string), material (string)
Example: {"price_max": 400, "style": "mid-century", "color": "brown"}
"""

async def parse_filters(user_query: str) -> dict:
    try:
        prompt = f"{SYSTEM_PROMPT}\n\nUser Query: {user_query}"
        
        
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0,
            )
        )
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Gemini API Error during parsing: {e}")
        return {}   