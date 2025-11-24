from typing import Dict, List, Any
import json
import random  # For mock scoring

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from modules import retrieval_modes, context_engineering, role_prompts, token_analysis
import config

class AdvancedRAG:
    """
    Orchestrates the RAG pipeline with corrective loops and quality checks.
    """
    
    def __init__(self, hybrid_retriever, graph_retriever, scratchpad):
        self.hybrid_retriever = hybrid_retriever
        self.graph_retriever = graph_retriever
        self.scratchpad = scratchpad
        self.context_eng = context_engineering.ContextEngineer()
        self.token_analyzer = token_analysis.TokenAnalyzer()
        # Keep temperature low for deterministic answers, but allow override via config later.
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL_NAME,
            temperature=0.2,
        )

    def evaluate_quality(self, context: List[Dict]) -> Dict[str, Any]:
        """
        Evaluates context quality (Mock logic).
        """
        # In production, use LLM to score relevance/completeness.
        score = random.uniform(0.4, 0.95) 
        
        label = "POOR"
        if score >= config.CONTEXT_QUALITY_THRESHOLDS["EXCELLENT"]:
            label = "EXCELLENT"
        elif score >= config.CONTEXT_QUALITY_THRESHOLDS["GOOD"]:
            label = "GOOD"
        elif score >= config.CONTEXT_QUALITY_THRESHOLDS["FAIR"]:
            label = "FAIR"
            
        return {"score": score, "label": label}

    def _format_context_block(self, chunks: List[Dict[str, Any]], limit: int = 5) -> str:
        """
        Builds a readable context string for LLM consumption.
        """
        if not chunks:
            return "No supporting documents were retrieved."

        lines = []
        for chunk in chunks[:limit]:
            source = chunk.get("source_filename", "Unknown Source")
            page = chunk.get("page", "N/A")
            text = chunk.get("text", "")
            lines.append(f"Source: {source} (Page {page})\n{text}")

        return "\n\n".join(lines)

    def _generate_llm_answer(self, query: str, selected_docs: List[Dict[str, Any]], role: str) -> Dict[str, Any]:
        """
        Calls the LLM to synthesize an answer from the selected context.
        """
        role_prompt = role_prompts.RolePrompts.get_prompt(role).strip()
        context_block = self._format_context_block(selected_docs)
        user_prompt = (
            "Use only the context below to answer the question. "
            "If the answer cannot be derived from the context, say you do not have enough information.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {query}"
        )

        prompt_tokens = self.token_analyzer.count_tokens(role_prompt + "\n\n" + user_prompt)

        final_prompt = f"{role_prompt}\n\n{user_prompt}"

        try:
            llm_response = self.llm.invoke(
                [
                    SystemMessage(content=role_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = getattr(llm_response, "content", str(llm_response))
            return {
                "answer_text": content,
                "raw_response": content,
                "prompt_tokens": prompt_tokens,
                "final_prompt": final_prompt,
            }
        except Exception as exc:
            fallback_text = (
                f"[Fallback Response] Unable to contact LLM: {exc}. "
                "Returning synthesized placeholder answer instead."
            )
            return {
                "answer_text": fallback_text,
                "raw_response": fallback_text,
                "prompt_tokens": prompt_tokens,
                "final_prompt": final_prompt,
            }

    def run_pipeline(self, query: str, mode: str, role: str = "Architect") -> Dict[str, Any]:
        """
        Executes the full RAG pipeline.
        """
        self.scratchpad.log(query, "Start", f"Started RAG pipeline in {mode} mode as {role}.")
        
        # 1. Retrieval
        retrieved_docs = []
        graph_context = {}
        
        tools_used = []

        if mode == retrieval_modes.RetrievalMode.TRADITIONAL.value or mode == retrieval_modes.RetrievalMode.HYBRID.value:
            retrieved_docs = self.hybrid_retriever.retrieve(query, k=config.RETRIEVAL_DEPTHS["MEDIUM"])
            self.scratchpad.log(query, "Retrieval", f"Retrieved {len(retrieved_docs)} docs via Hybrid.")
            tools_used.append("Vector Store (FAISS + Chroma)")

        if mode == retrieval_modes.RetrievalMode.KNOWLEDGE_GRAPH.value or mode == retrieval_modes.RetrievalMode.HYBRID.value:
            graph_context = self.graph_retriever.retrieve(query, k=5)
            self.scratchpad.log(query, "Retrieval", f"Retrieved graph context with {len(graph_context.get('nodes', []))} nodes.")
            tools_used.append("Knowledge Graph (Neo4j)")

        # 2. Context Engineering (Selection/Compression)
        # Convert docs to dicts if needed
        doc_dicts = [d.metadata for d in retrieved_docs]
        raw_tokens = sum(doc.get("tokens_estimate", 0) for doc in doc_dicts)
        raw_context_text = self._format_context_block(doc_dicts, limit=10)
        selection = self.context_eng.select_context(doc_dicts)
        selected_docs = selection["selected_chunks"]
        prepared_tokens = selection["total_tokens"]
        prepared_context_text = self._format_context_block(selected_docs, limit=10)
        
        # 3. Quality Check & Corrective Loop
        quality = self.evaluate_quality(selected_docs)
        self.scratchpad.log(query, "Evaluation", f"Context Quality: {quality['label']} ({quality['score']:.2f})")
        
        if quality["label"] in ["FAIR", "POOR"]:
            self.scratchpad.log(query, "Correction", "Quality low. Triggering corrective loop (Mock: Expanding query).")
            # Mock correction: just noting it. In real app, re-retrieve with expanded query.
            
        # 4. Answer Generation via LLM (with safe fallback)
        llm_outputs = self._generate_llm_answer(query, selected_docs, role)
        tools_used.append(f"LLM ({config.LLM_MODEL_NAME})")
        answer_text = llm_outputs["answer_text"]
        
        # 5. Structured Output
        response = {
            "answer": answer_text,
            "confidence": "High" if quality["score"] > 0.7 else "Medium",
            "sources": [d.get("source_filename") for d in selected_docs],
            "limitations": "Generated by a scaffold system. Verify with original docs.",
            "llm_output": llm_outputs["raw_response"],
            "tools_used": tools_used,
            "prompt_views": {
                "raw_context": raw_context_text,
                "prepared_context": prepared_context_text,
                "final_prompt": llm_outputs.get("final_prompt", ""),
            },
            "metrics": {
                "quality_score": quality["score"],
                "retrieval_mode": mode,
                "token_usage": {
                    "raw_context_tokens": raw_tokens,
                    "prepared_context_tokens": prepared_tokens,
                    "prompt_tokens": llm_outputs["prompt_tokens"],
                }
            }
        }
        
        self.scratchpad.log(query, "Completion", "Generated answer.", metadata=response)
        return response
