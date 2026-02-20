"""
Configuration for Buddha AI Application
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration."""

    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'buddha-ai-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True') == 'True'

    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL_CLASSIFIER = os.getenv('OPENAI_MODEL_CLASSIFIER', 'gpt-4o-mini')
    OPENAI_MODEL_GENERATOR = os.getenv('OPENAI_MODEL_GENERATOR', 'gpt-4o')

    # ChromaDB
    CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', 'data/chroma')

    # RAG Settings
    RAG_CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', '300'))
    RAG_CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP', '50'))
    RAG_N_RESULTS = int(os.getenv('RAG_N_RESULTS', '3'))

    # Embedding Model
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

    # Paths
    TEXTS_DIR = os.getenv('TEXTS_DIR', 'texts')
    STATIC_DIR = os.getenv('STATIC_DIR', 'static')
    IMAGES_DIR = os.path.join(STATIC_DIR, 'images')


# Validate critical configuration
def validate_config():
    """Validate that critical configuration is present."""
    if not Config.OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set in environment")
        print("Please add it to your .env file:")
        print("OPENAI_API_KEY=sk-...")
        return False
    return True


if __name__ == "__main__":
    print("Configuration:")
    print(f"  OpenAI API Key: {'Set' if Config.OPENAI_API_KEY else 'NOT SET'}")
    print(f"  Classifier Model: {Config.OPENAI_MODEL_CLASSIFIER}")
    print(f"  Generator Model: {Config.OPENAI_MODEL_GENERATOR}")
    print(f"  ChromaDB Path: {Config.CHROMA_PERSIST_DIR}")
    print(f"  Texts Directory: {Config.TEXTS_DIR}")
    print(f"  Valid: {validate_config()}")
