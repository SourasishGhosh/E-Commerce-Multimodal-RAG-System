import json
import logging

logger = logging.getLogger(__name__)

def safe_json_parse(json_str: str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}. Raw string: {json_str}")
        return {}

def normalize_vector(vector: list[float]):
    import math
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]