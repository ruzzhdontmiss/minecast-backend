import faiss
import os
import cv2
import librosa
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Setup paths
VIDEO_PATH = "sample_videos/video1.mp4"
TEMP_VIDEO_FILE = "debug_temp.mp4"

def test_cv2():
    print("Testing CV2...")
    try:
        with open(VIDEO_PATH, "rb") as f:
            video_bytes = f.read()
        
        with open(TEMP_VIDEO_FILE, "wb") as f:
            f.write(video_bytes)
        
        cap = cv2.VideoCapture(TEMP_VIDEO_FILE)
        if not cap.isOpened():
            print("Failed to open video")
            return
        
        ret, frame = cap.read()
        if ret:
            print(f"Read frame shape: {frame.shape}")
        cap.release()
        print("CV2 Test Parsed")
    except Exception as e:
        print(f"CV2 Exception: {e}")
    finally:
        if os.path.exists(TEMP_VIDEO_FILE):
            os.remove(TEMP_VIDEO_FILE)

def test_torch_clip():
    print("Testing Torch/CLIP...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create dummy image
        dummy_image = Image.new('RGB', (224, 224), color='red')
        inputs = processor(text=None, images=[dummy_image], return_tensors="pt", padding=True)
        img_features = model.get_image_features(inputs.pixel_values)
        print(f"CLIP Output fit: {img_features.shape}")
        print("Torch/CLIP Test Passed")
    except Exception as e:
        print(f"Torch/CLIP Exception: {e}")

def test_faiss():
    print("Testing FAISS...")
    try:
        d = 512
        index = faiss.IndexFlatIP(d)
        vec = np.random.rand(1, d).astype('float32')
        index.add(vec)
        print(f"FAISS index ntotal: {index.ntotal}")
        print("FAISS Test Passed")
    except Exception as e:
        print(f"FAISS Exception: {e}")

if __name__ == "__main__":
    print("Starting isolation tests...")
    test_cv2()
    test_torch_clip()
    try:
        test_faiss() # FAISS is often the culprit on Mac with OpenMP conflicts
    except:
        print("FAISS crashed hard (likely segfault)")
