# build_index.py

import os
import numpy as np
from video_processor import get_combined_embedding, EMBEDDING_DIM # This will now import EMBEDDING_DIM=512
import faiss

# --- Configuration ---
VIDEO_DIR = "sample_videos"
INDEX_FILE = "faiss_video.index" # The "database" file we will create

def build_and_save_index():
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Directory not found: {VIDEO_DIR}")
        return

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.mov'))]
    
    if not video_files:
        print(f"No videos found in {VIDEO_DIR}. Skipping index build.")
        return

    print(f"Found {len(video_files)} videos. Building FAISS index (VISUAL ONLY)...")

    # 1. Initialize the FAISS index
    # We use EMBEDDING_DIM (now 512)
    index = faiss.IndexFlatIP(EMBEDDING_DIM) 

    index_to_filename = []
    
    # 2. Process each video and add to index
    for filename in video_files:
        print(f"Processing: {filename}")
        video_path = os.path.join(VIDEO_DIR, filename)
        
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            
            embedding = get_combined_embedding(video_bytes)
            embedding_2d = np.array([embedding]).astype('float32')
            
            index.add(embedding_2d)
            index_to_filename.append(filename)
            
        except Exception as e:
            print(f"Could not process {filename}. Error: {e}")

    # 3. Save the index to disk
    print(f"Index build complete. Saving to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    
    # 4. Save the mapping
    with open("index_mapping.txt", "w") as f:
        for name in index_to_filename:
            f.write(f"{name}\n")
            
    print("--- Indexing Complete ---")
    print(f"Total videos indexed: {index.ntotal}")
    print(f"Index file: {INDEX_FILE}")
    print(f"Mapping file: index_mapping.txt")

# --- Run the script ---
if __name__ == "__main__":
    build_and_save_index()