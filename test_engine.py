import sys
import traceback
import numpy as np
import faiss

try:
    from ai_engine import AIEngine
    from video_processor import get_combined_embedding
    
    print("Initializing AIEngine...")
    engine = AIEngine()
    engine.load_model()
    
    with open("sample_videos/video1.mp4", "rb") as f:
        video_bytes = f.read()
        
    print("Generating embedding...")
    embedding = get_combined_embedding(video_bytes)
    print(f"Embedding generated. Shape: {embedding.shape}")
    
    print("Casting to float32...")
    embedding_2d = np.array([embedding]).astype('float32')
    embedding_2d = np.ascontiguousarray(embedding_2d, dtype=np.float32)
    print(f"embedding_2d shape: {embedding_2d.shape}")
    
    print("Normalizing L2...")
    faiss.normalize_L2(embedding_2d)
    print("Normalized.")
    
    print("Searching index...")
    D, I = engine.index.search(embedding_2d, 1)
    print(f"Searched. D={D}, I={I}")
    
except Exception as e:
    print(f"Exception: {e}")
    traceback.print_exc()
