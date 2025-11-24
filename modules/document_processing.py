import os
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document

import config

class DocumentProcessor:
    """
    Handles document ingestion, processing, and vector store management.
    """

    def __init__(self):
        """
        Initialize the processor with embedding model and text splitter.
        """
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
        
        # RecursiveCharacterTextSplitter is ideal for generic text as it tries to keep 
        # paragraphs, sentences, and words together.
        # Chunk size trade-off: 
        # - Larger chunks: More context, but potentially more noise and higher token cost.
        # - Smaller chunks: More precise retrieval, but might lose surrounding context.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""], # Priority: Paragraphs > Lines > Words > Chars
            length_function=len,
        )

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculates SHA256 hash of a file for caching purposes."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def process_pdf(self, file_path: str, original_filename: str) -> List[Dict[str, Any]]:
        """
        Loads a PDF, splits it into chunks, and returns metadata-rich documents.
        """
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Split text into chunks
        chunks = self.text_splitter.split_documents(pages)
        
        processed_chunks = []
        file_hash = self._calculate_file_hash(file_path)
        
        for i, chunk in enumerate(chunks):
            # Estimate tokens (rough approximation: 1 token ~= 4 chars)
            # For production, use tiktoken for exact counts.
            token_estimate = len(chunk.page_content) / 4 
            
            metadata = {
                "id": f"{file_hash}_{i}",
                "source_filename": original_filename,
                "page": chunk.metadata.get("page", 0) + 1, # 1-based indexing
                "text": chunk.page_content,
                "tokens_estimate": int(token_estimate),
                "created_at": datetime.now().isoformat(),
                "file_hash": file_hash
            }
            processed_chunks.append(metadata)
            
        return processed_chunks

    def update_vector_stores(self, processed_chunks: List[Dict[str, Any]]):
        """
        Updates both FAISS and Chroma vector stores with new chunks.
        """
        if not processed_chunks:
            return

        documents = [
            Document(page_content=chunk["text"], metadata=chunk) 
            for chunk in processed_chunks
        ]

        # 1. Update FAISS (Local, In-memory/File-based)
        # FAISS is great for fast similarity search but requires manual persistence handling usually.
        try:
            if os.path.exists(config.FAISS_INDEX_PATH):
                faiss_db = FAISS.load_local(
                    str(config.FAISS_INDEX_PATH), 
                    self.embeddings,
                    allow_dangerous_deserialization=True # Trusted local source
                )
                faiss_db.add_documents(documents)
            else:
                faiss_db = FAISS.from_documents(documents, self.embeddings)
            
            faiss_db.save_local(str(config.FAISS_INDEX_PATH))
        except Exception as e:
            print(f"Error updating FAISS: {e}")
            # Fallback or re-create logic could go here

        # 2. Update Chroma (Persistent, Database-like)
        # Chroma handles persistence automatically if persist_directory is set.
        try:
            Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(config.CHROMA_DB_PATH)
            )
        except Exception as e:
            print(f"Error updating Chroma: {e}")

    def reset_stores(self):
        """
        DANGER: Deletes all vector store data.
        """
        if config.VECTOR_STORE_DIR.exists():
            shutil.rmtree(config.VECTOR_STORE_DIR)
            config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            print("Vector stores reset successfully.")

# Example usage for testing
if __name__ == "__main__":
    # Mock config for standalone run
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("DocumentProcessor initialized.")
