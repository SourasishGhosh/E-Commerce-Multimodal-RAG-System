import open_clip     #type:ignore
import torch
from src.core.config import settings

class CLIPTextEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model, _, _ = open_clip.create_model_and_transforms(
            settings.clip_model_name,
            pretrained=settings.clip_pretrained
        )

        self.tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
        

        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> list[float]:
        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().tolist()