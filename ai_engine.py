import os
import faiss
import numpy as np
import shutil
from supabase import create_client, Client
from video_processor import get_combined_embedding, EMBEDDING_DIM

# Persistence Configuration
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY") 
# Note: In production, use SERVICE_ROLE_KEY for write access to storage if generic anon key doesn't have permissions. 
# For this demo/hackathon, we assume the provided key has necessary storage permissions or we rely on local FS if cloud fails.

BUCKET_NAME = "ai_models"
INDEX_FILENAME = "faiss_video.index"
MAPPING_FILENAME = "index_mapping.txt"

class AIEngine:
    def __init__(self):
        self.index = None
        self.index_to_filename = []
        self.supabase: Client = None
        
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                print("✅ Connected to Supabase for AI Model Persistence.")
            except Exception as e:
                print(f"⚠️ Failed to connect to Supabase: {e}")

    def load_model(self):
        """Loads the FAISS index from disk, downloading from Supabase first if available."""
        self._download_from_cloud()

        if not os.path.exists(INDEX_FILENAME):
            print(f"⚠️ No local index found at {INDEX_FILENAME}. Starting with empty index.")
            # Initialize empty index
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM) 
            self.index_to_filename = []
            return

        try:
            self.index = faiss.read_index(INDEX_FILENAME)
            
            if os.path.exists(MAPPING_FILENAME):
                with open(MAPPING_FILENAME, "r") as f:
                    self.index_to_filename = [line.strip() for line in f.readlines()]
            else:
                self.index_to_filename = []

            print(f"✅ AI Index Loaded. Total items: {self.index.ntotal}")

            if self.index.d != EMBEDDING_DIM:
                raise ValueError(f"Dimension Mismatch: Index {self.index.d} vs Model {EMBEDDING_DIM}")

        except Exception as e:
            print(f"❌ Error loading index: {e}")
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.index_to_filename = []

    def save_model(self):
        """Saves the index to disk and attempts to upload to Supabase."""
        if not self.index:
            return

        faiss.write_index(self.index, INDEX_FILENAME)
        with open(MAPPING_FILENAME, "w") as f:
            for name in self.index_to_filename:
                f.write(f"{name}\n")
        
        print(f"💾 Local Index Saved. Total items: {self.index.ntotal}")
        self._upload_to_cloud()

    def add_video(self, video_bytes: bytes, filename: str):
        """Processes a video and adds it to the index."""
        if self.index is None:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

        # 1. Generate Embedding
        embedding = get_combined_embedding(video_bytes)
        embedding_2d = np.array([embedding]).astype('float32')
        faiss.normalize_L2(embedding_2d)

        # 2. Add to Index
        self.index.add(embedding_2d)
        self.index_to_filename.append(filename)

        # 3. Save
        self.save_model()
        return self.index.ntotal

    def search(self, video_bytes: bytes, k=1):
        """Searches the index for the closest match."""
        if not self.index or self.index.ntotal == 0:
            return None, 0.0

        # 1. Generate Embedding
        embedding = get_combined_embedding(video_bytes)
        embedding_2d = np.array([embedding]).astype('float32')
        # Note: No normalization needed for search query in IndexFlatIP if vectors were normalized on add? 
        # Actually Inner Product usually requires normalization for Cosine Similarity equivalent.
        # video_processor.py's get_combined_embedding returns a vector. 
        # main.py didn't normalize query. Let's check consistency.
        # If we use normalize_L2 on add, we SHOULD normalize query for Cosine Similarity.
        # For safety/consistency with previous main.py (which didn't normalize add explicitly but video_processor might have?), 
        # let's look at previous main.py logic.
        # Previous main.py L186: faiss.normalize_L2(embedding_2d) ON ADD.
        # Previous main.py L95: NO normalize on query. This might be a subtle bug or assumed pre-normalized.
        # To be safe and correct for Cosine Similarity:
        faiss.normalize_L2(embedding_2d)

        D, I = self.index.search(embedding_2d, k)
        
        similarity = float(D[0][0])
        match_index = I[0][0]
        
        if match_index < 0 or match_index >= len(self.index_to_filename):
            match_filename = "Unknown"
        else:
            match_filename = self.index_to_filename[match_index]

        return match_filename, similarity

    def _download_from_cloud(self):
        if not self.supabase: return
        print("☁️ Attempting to download index from Supabase Storage...")
        try:
            # We can't easily "sync" files with python supabase client typically used for DB.
            # We need storage operations.
            # Using raw API or storage-py if available.
            # For simplicity in this environment, we'll try to just read bytes.
            
            # Downloading Index
            res_index = self.supabase.storage.from_(BUCKET_NAME).download(INDEX_FILENAME)
            with open(INDEX_FILENAME, "wb") as f:
                f.write(res_index)
            
            # Downloading Mapping
            res_map = self.supabase.storage.from_(BUCKET_NAME).download(MAPPING_FILENAME)
            with open(MAPPING_FILENAME, "wb") as f:
                f.write(res_map)
                
            print("✅ Downloaded Index from Cloud.")
        except Exception as e:
            print(f"ℹ️ Could not download from cloud (Bucket might be empty or missing): {e}")

    def _upload_to_cloud(self):
        if not self.supabase: return
        print("☁️ Uploading updated index to Supabase Storage...")
        try:
            # Upload Index
            with open(INDEX_FILENAME, "rb") as f:
                self.supabase.storage.from_(BUCKET_NAME).upload(
                    INDEX_FILENAME, 
                    f, 
                    file_options={"upsert": "true"}
                )

            # Upload Mapping
            with open(MAPPING_FILENAME, "rb") as f:
                self.supabase.storage.from_(BUCKET_NAME).upload(
                    MAPPING_FILENAME, 
                    f, 
                    file_options={"upsert": "true"}
                )
            print("✅ Uploaded Index to Cloud.")
        except Exception as e:
            print(f"❌ Failed to upload to cloud: {e}")
