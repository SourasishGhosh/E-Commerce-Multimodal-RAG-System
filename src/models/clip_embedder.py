import open_clip #type:ignore
import torch
from PIL import Image
from io import BytesIO
from src.core.config import settings

class CLIPEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name,
            pretrained=settings.clip_pretrained
        )
        
        self.model.to(self.device)
        self.model.eval()

    def embed_image(self, image_path: str):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().tolist() 

    def embed_image_bytes(self, image_bytes: bytes):
        image = self.preprocess(Image.open(BytesIO(image_bytes))).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze().cpu().tolist()