from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "products"
    
    gemini_api_key: str = ""   
    
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    embedding_dim: int = 512

    class Config:
        env_file = ".env"

settings = Settings()