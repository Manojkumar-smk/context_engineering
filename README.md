# Advanced Multi-Agent RAG System

A modular Streamlit-based RAG system scaffold designed for advanced document processing, retrieval, and agentic workflows.

## Features
- **Document Processing**: PDF upload, semantic chunking, and metadata extraction.
- **Dual Vector Stores**: FAISS (fast local) and Chroma (persistent).
- **Knowledge Graph**: Neo4j integration for graph-based retrieval.
- **Advanced Retrieval**: Hybrid (Vector + Graph) and Corrective RAG loops.
- **Analysis Tools**: Token usage tracking, cost estimation, and scratchpad.
- **Role-Based Agents**: Configurable agent personas (Architect, Developer, Analyst).

## Setup

### Prerequisites
- Python 3.11+
- Docker (optional)
- OpenAI API Key
- Neo4j Instance (Local or AuraDB)

### Local Installation
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   Create a `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=password
   ```
5. Run the application:
   ```bash
   streamlit run app.py
   ```

### Docker
```bash
docker build -t advanced-rag .
docker run -p 8501:8501 --env-file .env advanced-rag
```

## Neo4j Setup
1. **Local**: Run `docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest`
2. **Remote**: Use Neo4j AuraDB and update `NEO4J_URI` in `.env`.

## Usage Flow
1. **Upload**: Drag & drop PDFs in the "Document Ingestion" section.
2. **Configure**: Select "Hybrid" or "Knowledge Graph" mode and an Agent Role.
3. **Query**: Ask a question. The system will retrieve context, evaluate quality, and generate a structured answer.
4. **Analyze**: Check the "Token Usage Analysis" chart and "Scratchpad" for insights.

## Limitations
- **Graph Ingestion**: Currently uses a heuristic (capitalized words) to create entities. Production systems should use NER models.
- **Compression**: Map-reduce summarization is a placeholder.
- **Corrective Loop**: Mocked for demonstration; requires real recursive calls in production.
# context_engineering
