from enum import Enum
from typing import List, Dict, Any, Optional
import os
import json

from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from neo4j import GraphDatabase

import config

class RetrievalMode(Enum):
    TRADITIONAL = "Traditional (FAISS+Chroma)"
    KNOWLEDGE_GRAPH = "Knowledge Graph (Neo4j)"
    HYBRID = "Hybrid (Vector + Graph)"

def get_retrieval_mode_description(mode: RetrievalMode) -> str:
    descriptions = {
        RetrievalMode.TRADITIONAL: "Vectors for semantic similarity. Combines FAISS (fast) and Chroma (persistent).",
        RetrievalMode.KNOWLEDGE_GRAPH: "Graph for explicit relationships. Uses Neo4j to find connected entities.",
        RetrievalMode.HYBRID: "Combines semantic vectors and graph relationships for maximum context."
    }
    return descriptions.get(mode, "Unknown mode")

class HybridRetriever:
    """
    Combines results from FAISS and Chroma with de-duplication and diversity.
    """
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.faiss_store = None
        self.chroma_store = None
        self._load_stores()

    def _load_stores(self):
        if os.path.exists(config.FAISS_INDEX_PATH):
            try:
                self.faiss_store = FAISS.load_local(
                    str(config.FAISS_INDEX_PATH), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load FAISS: {e}")

        try:
            self.chroma_store = Chroma(
                persist_directory=str(config.CHROMA_DB_PATH), 
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"Failed to load Chroma: {e}")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        results = []
        
        # 1. Query FAISS (MMR for diversity)
        if self.faiss_store:
            try:
                faiss_docs = self.faiss_store.max_marginal_relevance_search(query, k=k, fetch_k=k*2)
                results.extend(faiss_docs)
            except Exception as e:
                print(f"FAISS retrieval error: {e}")

        # 2. Query Chroma (Similarity)
        if self.chroma_store:
            try:
                chroma_docs = self.chroma_store.similarity_search(query, k=k)
                results.extend(chroma_docs)
            except Exception as e:
                print(f"Chroma retrieval error: {e}")

        # 3. Merge & Deduplicate (by chunk ID or content hash)
        unique_docs = {}
        for doc in results:
            # Prefer ID if available, else hash content
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
        
        return list(unique_docs.values())[:k]

class GraphRetriever:
    """
    Handles Neo4j interactions for Knowledge Graph retrieval.
    """
    def __init__(self):
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(
                config.NEO4J_URI, 
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
        except Exception as e:
            print(f"Neo4j connection failed: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def ingest_minimal(self, documents: List[Dict[str, Any]]):
        """
        Ingests basic entity/relationship data from documents.
        This is a simplified heuristic ingestion.
        """
        if not self.driver:
            return

        with self.driver.session() as session:
            for doc in documents:
                # Create Document Node
                session.run(
                    """
                    MERGE (d:Document {id: $id})
                    SET d.filename = $filename, d.page = $page
                    """,
                    id=doc["id"], filename=doc["source_filename"], page=doc["page"]
                )
                
                # Heuristic: Extract capitalized words as potential "Entities" (Very naive)
                # In production, use an LLM or NER model here.
                text = doc["text"]
                words = set([w.strip(".,") for w in text.split() if w[0].isupper() and len(w) > 3])
                
                for word in list(words)[:5]: # Limit to top 5 to avoid noise
                    session.run(
                        """
                        MATCH (d:Document {id: $id})
                        MERGE (e:Entity {name: $name})
                        MERGE (d)-[:MENTIONS]->(e)
                        """,
                        id=doc["id"], name=word
                    )

    def retrieve(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieves a subgraph context based on the query.
        """
        if not self.driver:
            return {"nodes": [], "edges": [], "text": "Graph DB not connected."}

        # 1. Identify potential entities in query (Naive)
        query_entities = [w.strip(".,") for w in query.split() if w[0].isupper()]
        
        context_bundle = {"nodes": [], "edges": [], "supporting_texts": []}
        
        with self.driver.session() as session:
            for entity in query_entities:
                result = session.run(
                    """
                    MATCH (e:Entity {name: $name})<-[r:MENTIONS]-(d:Document)
                    RETURN e, r, d
                    LIMIT $k
                    """,
                    name=entity, k=k
                )
                
                for record in result:
                    e_node = record["e"]
                    d_node = record["d"]
                    context_bundle["nodes"].append(dict(e_node))
                    context_bundle["nodes"].append(dict(d_node))
                    context_bundle["supporting_texts"].append(f"Document {d_node.get('filename')} mentions {e_node.get('name')}")

        return context_bundle
