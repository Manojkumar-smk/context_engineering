import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index"
CHROMA_DB_PATH = VECTOR_STORE_DIR / "chroma_db"
SCRATCHPAD_PATH = DATA_DIR / "scratchpad.json"
TEMP_UPLOAD_DIR = DATA_DIR / "temp_uploads"

# Ensure directories exist
for path in [DATA_DIR, VECTOR_STORE_DIR, TEMP_UPLOAD_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# --- Models ---
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
LLM_MODEL_NAME = "gpt-4o"

# --- Pricing (USD per 1M tokens) ---
# Approximate pricing as of late 2024
PRICING = {
    "gpt-4o": {
        "input": 5.00,
        "output": 15.00
    },
    "text-embedding-3-large": {
        "input": 0.13,
        "output": 0.00
    }
}

# --- Chunking Settings ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# --- Neo4j Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Please check your .env file.")

# --- RAG Settings ---
CONTEXT_QUALITY_THRESHOLDS = {
    "EXCELLENT": 0.8,
    "GOOD": 0.65,
    "FAIR": 0.5
}

RETRIEVAL_DEPTHS = {
    "SHALLOW": 5,
    "MEDIUM": 10,
    "DEEP": 20
}

# --- App Settings ---
APP_TITLE = "Advanced Multi-Agent RAG"
APP_ICON = "🦅"
