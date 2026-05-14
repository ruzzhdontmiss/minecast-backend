# video_processor.py

import cv2
import librosa
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

# --- DISABLE TENSORFLOW/YAMNET ---
# import tensorflow as tf
# import tensorflow_hub as hub
# ----------------------------------

# --- Constants ---
FRAME_COUNT = 5
AUDIO_DURATION = 5.0

# --- CHANGED EMBEDDING_DIM ---
# We are ONLY using CLIP (512) and disabling YAMNet (1024)
EMBEDDING_DIM = 512 
# -------------------------------

TEMP_VIDEO_FILE = "temp_video_for_cv.mp4"
TEMP_AUDIO_FILE = "temp_audio_for_librosa.mp4"

# --- AI Model Loading ---
print("Loading AI models (Visual Only)...")
# 1. CLIP (Visual) Model
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# --- DISABLE TENSORFLOW/YAMNET ---
# 2. YAMNet (Audio) Model
# tf.get_logger().setLevel('ERROR')
# yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
# ----------------------------------
print("AI models loaded successfully.")

# --- Helper Functions ---

import tempfile

def get_frames_from_video(video_bytes: bytes):
    cap = None
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(video_bytes)
            temp_path = f.name
        
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            print(f"Error opening video file: {temp_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        
        if total_frames == 0:
             print(f"Video file has 0 frames: {temp_path}")
             return []
        elif total_frames < FRAME_COUNT:
            indices = np.arange(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, FRAME_COUNT, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        
        cap.release()
        return frames
    
    except Exception as e:
        print(f"Error in get_frames_from_video: {e}")
        if cap:
            cap.release()
        return []
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# --- This function is no longer called, but we leave it ---
def get_audio_from_video(video_bytes: bytes):
    try:
        with open(TEMP_AUDIO_FILE, "wb") as f:
            f.write(video_bytes)
        waveform, sample_rate = librosa.load(TEMP_AUDIO_FILE, sr=None, offset=0.0, duration=AUDIO_DURATION)
        if waveform.ndim > 1:
            waveform = librosa.to_mono(waveform)
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        return waveform, sample_rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        return np.zeros(int(16000 * AUDIO_DURATION)), 16000
    finally:
        if os.path.exists(TEMP_AUDIO_FILE):
            os.remove(TEMP_AUDIO_FILE)

def get_visual_embedding(frames: list):
    if not frames:
        return np.zeros(512)
    with torch.no_grad():
        inputs = clip_processor(text=None, images=frames, return_tensors="pt", padding=True)
        image_features = clip_model.get_image_features(inputs.pixel_values)
        avg_features = torch.mean(image_features, dim=0)
        return avg_features.numpy().flatten()

# --- This function is no longer called ---
def get_audio_embedding(waveform: np.ndarray):
    return np.zeros(1024) # Return a dummy array

def get_combined_embedding(video_bytes: bytes):
    """
    The main function to get a VISUAL-ONLY embedding for a video.
    """
    # 1. Process Video
    frames = get_frames_from_video(video_bytes)
    visual_emb = get_visual_embedding(frames)
    
    # --- DISABLE AUDIO ---
    # 2. Process Audio
    # waveform, _ = get_audio_from_video(video_bytes)
    # audio_emb = get_audio_embedding(waveform)
    # ---------------------
    
    # --- CHANGED EMBEDDING ---
    # 3. Combine them (Now just visual)
    combined_emb = visual_emb.astype('float32')
    # -------------------------
    
    # 4. Normalize
    norm = np.linalg.norm(combined_emb)
    if norm == 0:
        return combined_emb
    return combined_emb / norm