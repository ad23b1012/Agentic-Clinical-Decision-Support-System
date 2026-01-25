"""
reset_vectors.py
----------------
WARNING: Deletes ALL vectors in your Pinecone Index.
"""
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_URL = os.getenv("PINECONE_INDEX_URL")
    
if not PINECONE_API_KEY: raise RuntimeError("Missing PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_URL)

def wipe():
    print(f"[INFO] Connecting to {PINECONE_INDEX_URL}...")
    stats = index.describe_index_stats()
    print(f"[INFO] Current Vector Count: {stats.total_vector_count}")
    
    print("\n[WARNING] DELETING ALL VECTORS in 5 seconds... (Ctrl+C to cancel)")
    time.sleep(5)
    
    try:
        index.delete(delete_all=True)
        print("[SUCCESS] All vectors deleted.")
    except Exception as e:
        print(f"[ERROR] Delete failed: {e}")
        
    time.sleep(2)
    print(f"[INFO] New Vector Count: {index.describe_index_stats().total_vector_count}")

if __name__ == "__main__":
    wipe()