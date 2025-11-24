import json
import os
from datetime import datetime
from typing import List, Dict, Any
import config

class Scratchpad:
    """
    Manages a persistent scratchpad for the user with structured logging.
    """
    def __init__(self):
        self.path = config.SCRATCHPAD_PATH
        self._ensure_file()
        
    def _ensure_file(self):
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump([], f)
                
    def load(self, limit: int = 50) -> List[Dict]:
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
                return data[-limit:] # Return last N entries
        except:
            return []
            
    def log(self, query: str, step: str, content: str, metadata: Dict = None):
        """
        Logs an event to the scratchpad.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "step": step, # e.g., "Retrieval", "Selection", "Correction"
            "content": content,
            "metadata": metadata or {}
        }
        
        entries = []
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                try:
                    entries = json.load(f)
                except:
                    pass
        
        entries.append(entry)
        
        with open(self.path, "w") as f:
            json.dump(entries, f, indent=2)

    def clear(self):
        with open(self.path, "w") as f:
            json.dump([], f)

class ContextEngineer:
    """
    Handles context selection, compression, and isolation.
    """
    
    def __init__(self, token_analyzer=None):
        self.token_analyzer = token_analyzer

    def select_context(self, chunks: List[Dict], max_tokens: int = 3000) -> Dict[str, Any]:
        """
        Selects top chunks based on relevance/diversity up to max_tokens.
        """
        selected = []
        current_tokens = 0
        
        # Sort by some score if available, else assume retrieval order is ranked
        # In a real system, we'd re-rank here using a Cross-Encoder.
        
        for chunk in chunks:
            tokens = chunk.get("tokens_estimate", 0)
            if current_tokens + tokens > max_tokens:
                break
            
            selected.append(chunk)
            current_tokens += tokens
            
        return {
            "selected_chunks": selected,
            "total_tokens": current_tokens,
            "dropped_count": len(chunks) - len(selected)
        }

    def compress_context(self, chunks: List[Dict]) -> str:
        """
        Compresses chunks into a coherent summary (Map-Reduce placeholder).
        """
        # Placeholder for LLM-based compression
        # In production: Map (summarize each) -> Reduce (combine)
        
        summary_lines = []
        for i, chunk in enumerate(chunks):
            # Naive compression: First sentence + source
            text = chunk.get("text", "")
            first_sentence = text.split(".")[0] + "."
            source = f"[{chunk.get('source_filename')}:{chunk.get('page')}]"
            summary_lines.append(f"- {first_sentence} {source}")
            
        return "\n".join(summary_lines)

    def isolate_subtask(self, task_name: str, input_data: Any) -> Dict:
        """
        Sandboxes a subtask (e.g., query classification).
        """
        # Placeholder for subtask logic
        return {
            "task": task_name,
            "status": "completed",
            "result": f"Processed {task_name}"
        }
