from video_processor import get_frames_from_video, get_visual_embedding

with open("sample_videos/video1.mp4", "rb") as f:
    video_bytes = f.read()

print("get_frames_from_video")
frames = get_frames_from_video(video_bytes)
print(f"Frames: {len(frames)}")

print("get_visual_embedding")
emb = get_visual_embedding(frames)
print(f"Visual embedding: {emb.shape}")
