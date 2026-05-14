# main.py
import os
import random

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="MineCast AI Engine (Lightweight)",
    description="Analyzes video originality and returns a rarity score. (MOCK FOR FREE TIER)",
    version="0.3.0",
)

# TODO: re-enable CLIP/FAISS on paid tier

# CORS
origins = ["http://localhost:3000", "http://localhost", "https://minecast.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "MineCast AI Engine Online (Lightweight Mode)",
        "index_status": "Mocked",
        "items_in_index": 0
    }

import hashlib

UPLOADED_HASHES = set()

@app.post("/analyze")
async def analyze_video(video_file: UploadFile = File(...)):
    print(f"Received for analysis (HASH): {video_file.filename}")
    video_bytes = await video_file.read()
    
    # Calculate SHA-256 hash of the video file
    file_hash = hashlib.sha256(video_bytes).hexdigest()
    
    if file_hash in UPLOADED_HASHES:
        # Duplicate detected
        score = random.uniform(0.05, 0.15)
        is_unique = False
        closest_match = f"hash:{file_hash[:8]}..."
    else:
        # Original video
        score = random.uniform(0.75, 0.99)
        is_unique = True
        closest_match = "None (Original)"
        
    return {
        "filename": video_file.filename,
        "is_unique": is_unique,
        "rarity_score": round(float(score), 4),
        "debug_info": {
            "closest_match_filename": closest_match,
            "similarity_score": round(float(1.0 - score), 4)
        }
    }

@app.post("/add_video")
async def add_video(video_file: UploadFile = File(...)):
    print(f"Adding to index (HASH): {video_file.filename}")
    video_bytes = await video_file.read()
    file_hash = hashlib.sha256(video_bytes).hexdigest()
    UPLOADED_HASHES.add(file_hash)
    
    return {"message": "Video added to hash index", "total_videos": len(UPLOADED_HASHES)}

from pydantic import BaseModel
from web3 import Web3
from eth_account.messages import encode_defunct

class ClaimRequest(BaseModel):
    wallet_address: str
    video_id: str

@app.post("/claim-reward")
async def claim_reward(req: ClaimRequest):
    print(f"Processing claim for wallet: {req.wallet_address}, video: {req.video_id}")
    
    private_key = os.getenv("SIGNER_PRIVATE_KEY")
    contract_address = os.getenv("TREASURY_CONTRACT_ADDRESS")
    
    if not private_key or not contract_address:
        print("Missing SIGNER_PRIVATE_KEY or TREASURY_CONTRACT_ADDRESS in .env")
        raise HTTPException(status_code=500, detail="Server missing Web3 configuration")

    try:
        wallet = Web3.to_checksum_address(req.wallet_address)
        contract = Web3.to_checksum_address(contract_address)
        
        message_hash = Web3.solidity_keccak(
            ['address', 'string', 'address'],
            [wallet, req.video_id, contract]
        )
        
        signable_message = encode_defunct(message_hash)
        
        from eth_account import Account
        signed_message = Account.sign_message(signable_message, private_key=private_key)
        
        return {
            "signature": signed_message.signature.hex(),
            "video_id": req.video_id,
            "wallet_address": req.wallet_address
        }
    except Exception as e:
        print(f"Error generating signature: {e}")
        raise HTTPException(status_code=500, detail=str(e))