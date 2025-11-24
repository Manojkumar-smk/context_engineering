import json
import os
from datetime import datetime
import config

class Scratchpad:
    """
    Manages a persistent scratchpad for the user.
    """
    
    def __init__(self):
        self.path = config.SCRATCHPAD_PATH
        self._ensure_file()
        
    def _ensure_file(self):
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)
                
    def load(self):
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_entries(self, entries):
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2)
            
    def add_entry(self, content: str, source: str = "User"):
        entries = self.load()
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "content": content
        }
        entries.append(entry)
        self._write_entries(entries)
            
    def clear(self):
        self._write_entries([])

    def log(self, query: str, step: str, message: str, metadata: dict = None):
        """
        Logs a step in the RAG pipeline.
        """
        entries = self.load()
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source": "System",
            "query": query,
            "step": step,
            "content": message,
            "metadata": metadata or {}
        }
        entries.append(entry)
        self._write_entries(entries)
