import pytest
from src.models.clip_embedder import CLIPEmbedder
from src.models.text_embedder import CLIPTextEmbedder

@pytest.fixture(scope="module")
def embedders():
    return CLIPEmbedder(), CLIPTextEmbedder()

def test_image_and_text_same_dimension(embedders, tmp_path):
    img_emb, txt_emb = embedders
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(img_path)

    img_vec = img_emb.embed_image(str(img_path))
    txt_vec = txt_emb.embed_text("a grey square")

    # This is the critical invariant: both must be 512-dim (ViT-B-32)
    assert len(img_vec) == len(txt_vec) == 512, (
        "Image and text vectors must share the same dimension to be comparable."
    )

def test_vectors_are_normalized(embedders, tmp_path):
    import numpy as np
    img_emb, txt_emb = embedders
    txt_vec = txt_emb.embed_text("wooden mid-century coffee table")
    norm = np.linalg.norm(txt_vec)
    assert abs(norm - 1.0) < 1e-5, "Vectors must be L2-normalized before storing."