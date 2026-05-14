# main.py
import os
os.environ["OMP_NUM_THREADS"] = "1" # Fix for FAISS/PyTorch OpenMP segfault

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ai_engine import AIEngine
from dotenv import load_dotenv

# Load env from parent directory (where .env.local usually is for Next.js, 
# but for python we might need to be explicit if it's not in current dir)
load_dotenv()

app = FastAPI(
    title="MineCast AI Engine",
    description="Analyzes video originality and returns a rarity score.",
    version="0.2.0 (Hardened)",
)

# Initialize Engine
engine = AIEngine()

@app.on_event("startup")
def startup_event():
    engine.load_model()

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
    status = "Active" if engine.index and engine.index.ntotal > 0 else "Initializing/Empty"
    return {
        "message": "MineCast AI Engine Online",
        "index_status": status,
        "items_in_index": engine.index.ntotal if engine.index else 0
    }

@app.post("/analyze")
async def analyze_video(video_file: UploadFile = File(...)):
    print(f"Received for analysis: {video_file.filename}")
    video_bytes = await video_file.read()
    
    try:
        closest_match, similarity = engine.search(video_bytes)
        
        # Rarity Logic
        COPY_THRESHOLD = 0.90
        if similarity > COPY_THRESHOLD:
            # Copy
            rarity_score = 1.0 - similarity 
            is_unique = False
        else:
            # Unique: Scale remaining 0-0.9 to 0.1-1.0
            rarity_score = (1.0 - (similarity / COPY_THRESHOLD)) * 0.9 + 0.1
            is_unique = True
            
        return {
            "filename": video_file.filename,
            "is_unique": is_unique,
            "rarity_score": round(float(rarity_score), 4),
            "debug_info": {
                "closest_match_filename": closest_match,
                "similarity_score": round(float(similarity), 4)
            }
        }
    except Exception as e:
        print(f"Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_video")
async def add_video(video_file: UploadFile = File(...)):
    print(f"Adding to index: {video_file.filename}")
    video_bytes = await video_file.read()
    
    try:
        count = engine.add_video(video_bytes, video_file.filename)
        return {"message": "Video added", "total_videos": count}
    except Exception as e:
        print(f"Error adding video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel
import os
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
        # In a real app we'd raise 500, but for local testing without proper env we can fallback or raise.
        raise HTTPException(status_code=500, detail="Server missing Web3 configuration")

    # Hash the payload exactly as Solidity would: keccak256(abi.encodePacked(address, string, address))
    try:
        # Convert wallet and contract to checksum addresses
        wallet = Web3.to_checksum_address(req.wallet_address)
        contract = Web3.to_checksum_address(contract_address)
        
        message_hash = Web3.solidity_keccak(
            ['address', 'string', 'address'],
            [wallet, req.video_id, contract]
        )
        
        signable_message = encode_defunct(message_hash)
        
        # We need eth_account to sign it
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