import os
import math
import uuid
import json
import pandas as pd
import requests
from tqdm import tqdm
from qdrant_client import QdrantClient  #type:ignore
from qdrant_client.models import PointStruct, VectorParams, Distance  #type:ignore
from src.models.clip_embedder import CLIPEmbedder
from src.core.config import settings

def init_qdrant_collection(client: QdrantClient):
    
    existing = client.get_collections().collections
    collection_names = [c.name for c in existing]
    
    if settings.qdrant_collection not in collection_names:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
        print(f"Collection '{settings.qdrant_collection}' created.")
    else:
        print(f"Collection '{settings.qdrant_collection}' already exists.")

def ingest_data(csv_path: str):
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    init_qdrant_collection(client)
    embedder = CLIPEmbedder()
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=100000)
    points = []
    
    for index, row in df.iterrows():
        try:
            img_url = str(row.get('imgUrl', '')).strip()
            
            if not img_url or img_url.lower() == 'nan' or not img_url.startswith('http'):
                print(f" Skipping row {index}: Invalid URL. Found: '{img_url}'")
                continue
            try:
                response = requests.get(img_url, timeout=5)
                response.raise_for_status() 
                image_bytes = response.content
            except requests.exceptions.RequestException as e:
                print(f" Skipping row {index}: Could not download image. Error: {e}")
                continue

            payload = {
                "asin": str(row.get('asin', '')),
                "product_name": str(row.get('title', 'Unknown')),
                "image_url": img_url,
                "product_url": str(row.get('productURL', '')),
                "price": float(row.get('price', 0.0)) if not pd.isna(row.get('price')) else 0.0,
                "stars": float(row.get('stars', 0.0)) if not pd.isna(row.get('stars')) else 0.0,
                "reviews": int(row.get('reviews', 0)) if not pd.isna(row.get('reviews')) else 0,
                "category": str(row.get('category_id', '')),
                "is_best_seller": bool(row.get('isBestSeller', False))
            }
            
            
            vector = embedder.embed_image_bytes(image_bytes)
            
            
            asin_val = str(row.get('asin', str(uuid.uuid4())))
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, asin_val))
            
            points.append(
                PointStruct(id=point_id, vector=vector, payload=payload)
            )
            
            if len(points) >= 100:
                client.upsert(collection_name=settings.qdrant_collection, points=points)
                points = []
                print(f"Upserted batch ending at index {index}")
                
        except Exception as e:
            print(f"Error processing row {index}: {type(e).__name__} - {e}")
            
    if points:
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        print(" Final batch upserted.")

if __name__ == "__main__":
    
    ingest_data("data/amazon_products/amazon_products.csv")