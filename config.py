import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

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

# --- Configuration Helper Functions ---
def get_config_value(key: str, default: Optional[str] = None, user_config: Optional[dict] = None) -> Optional[str]:
    """
    Get configuration value with priority: user_config > .env > default
    
    Args:
        key: Environment variable name
        default: Default value
        user_config: Dictionary of user-provided config values (from session state)
    
    Returns:
        Configuration value or default
    """
    # Priority 1: User-provided config (from Streamlit UI)
    if user_config and key in user_config and user_config[key]:
        return user_config[key]
    
    # Priority 2: Environment variable from .env file
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    # Priority 3: Default value
    return default

# --- Neo4j Configuration ---
# These will be set dynamically based on user input
NEO4J_URI_DEFAULT = "bolt://localhost:7687"
NEO4J_USER_DEFAULT = "neo4j"
NEO4J_PASSWORD_DEFAULT = "password"

def get_neo4j_config(user_config: Optional[dict] = None):
    """Get Neo4j configuration values"""
    return {
        "uri": get_config_value("NEO4J_URI", NEO4J_URI_DEFAULT, user_config),
        "user": get_config_value("NEO4J_USER", NEO4J_USER_DEFAULT, user_config),
        "password": get_config_value("NEO4J_PASSWORD", NEO4J_PASSWORD_DEFAULT, user_config)
    }

# --- OpenAI Configuration ---
def get_openai_api_key(user_config: Optional[dict] = None) -> Optional[str]:
    """Get OpenAI API key from user config or .env"""
    return get_config_value("OPENAI_API_KEY", None, user_config)

# For backward compatibility, set defaults (will be overridden when user config is provided)
NEO4J_URI = os.getenv("NEO4J_URI", NEO4J_URI_DEFAULT)
NEO4J_USER = os.getenv("NEO4J_USER", NEO4J_USER_DEFAULT)
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", NEO4J_PASSWORD_DEFAULT)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Note: OPENAI_API_KEY validation is now done in the app when user config is provided

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
