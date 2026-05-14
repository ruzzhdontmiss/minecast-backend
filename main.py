from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from web3 import Web3
from eth_account.messages import encode_defunct
import hashlib
import random
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_hashes = set()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    file_hash = hashlib.md5(contents).hexdigest()
    
    if file_hash in video_hashes:
        score = random.randint(5, 15)
        tier = "Common"
        is_duplicate = True
    else:
        video_hashes.add(file_hash)
        score = random.randint(75, 99)
        tier = "Legendary" if score > 90 else "Rare"
        is_duplicate = False
    
    return {
        "rarity_score": score,
        "tier": tier,
        "is_duplicate": is_duplicate,
        "message": "Duplicate detected" if is_duplicate else "Original content"
    }

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