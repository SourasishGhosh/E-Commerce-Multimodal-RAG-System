import google.generativeai as genai  #type:ignore
from src.core.config import settings
from src.utils.helpers import safe_json_parse

# Configure the Gemini SDK
genai.configure(api_key=settings.gemini_api_key)

# Gemini 1.5 Flash is the industry standard for fast, cheap text extraction
model = genai.GenerativeModel('gemini-1.5-flash')

SYSTEM_PROMPT = """
You are an e-commerce metadata extractor. Analyze the product description and extract key attributes.
Return ONLY a valid JSON object. 
Keys allowed: color (string), material (string), style (string), price (number).
If an attribute is not mentioned, omit the key.
"""

def extract_metadata(description: str) -> dict:
    if not description or not isinstance(description, str):
        return {}

    try:
        # We combine the system prompt and the user input
        prompt = f"{SYSTEM_PROMPT}\n\nDescription: {description}"
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json", # Enforces JSON output
                temperature=0,
            )
        )
        return safe_json_parse(response.text)
    except Exception as e:
        print(f"Failed to extract metadata: {e}")
        return {}