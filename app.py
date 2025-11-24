import streamlit as st
import os
import tempfile
from modules import ui_components, document_processing, retrieval_modes, token_analysis, scratchpad_db, advanced_rag, context_engineering, role_prompts, langgraph_visual
import config

# Initialize modules
doc_processor = document_processing.DocumentProcessor()
token_analyzer = token_analysis.TokenAnalyzer()
pad = scratchpad_db.Scratchpad()
hybrid_retriever = retrieval_modes.HybridRetriever(doc_processor.embeddings)
graph_retriever = retrieval_modes.GraphRetriever()
rag_engine = advanced_rag.AdvancedRAG(hybrid_retriever, graph_retriever, pad)

def main():
    ui_components.render_header()
    ingestion_mode, selected_mode, selected_role = ui_components.render_sidebar(
        retrieval_modes.RetrievalMode, 
        role_prompts.RolePrompts
    )

    # --- Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1. Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Upload one or more PDFs to build your knowledge base."
        )

        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            chunks = doc_processor.process_pdf(tmp_path, uploaded_file.name)
                            all_chunks.extend(chunks)
                        finally:
                            os.remove(tmp_path)
                    
                    if ingestion_mode in ["Traditional RAG (Vector Store)", "Both"]:
                        doc_processor.update_vector_stores(all_chunks)
                    
                    if ingestion_mode in ["Knowledge Graph", "Both"]:
                        graph_retriever.ingest_minimal(all_chunks) # Ingest to Graph
                    
                    ui_components.render_file_upload_stats(uploaded_files, all_chunks)

        st.markdown("---")
        st.subheader("2. Query & Retrieval")
        
        # Controls moved to sidebar

        query = st.text_area("Enter your query:", height=100)
        
        if st.button("Ask Agent"):
            if not query:
                st.warning("Please enter a query.")
            else:
                with st.spinner("Running Advanced RAG Pipeline..."):
                    # Visualize Workflow
                    langgraph_visual.LangGraphVisualizer.render_graph(current_step="Answer")
                    
                    # Run Pipeline
                    response = rag_engine.run_pipeline(query, selected_mode, selected_role)
                    
                    # --- NEW: Tabs for structured vs raw response ---
                    tab1, tab2 = st.tabs(["Structured Answer", "Raw Response"])
                    with tab1:
                        ui_components.render_structured_answer(response)
                    with tab2:
                        st.markdown("### LLM Output")
                        llm_output = response.get("llm_output") or response.get("answer")
                        st.write(llm_output)
                        st.markdown("### Full Response Payload")
                        st.json(response)

                    # Token Analysis using real metrics when available
                    token_metrics = response.get("metrics", {}).get("token_usage", {})
                    raw_tokens = token_metrics.get("raw_context_tokens", 0)
                    prepared_tokens = token_metrics.get("prepared_context_tokens", 0)
                    prompt_tokens = token_metrics.get("prompt_tokens", 0)
                    
                    if raw_tokens == 0 and prepared_tokens == 0 and prompt_tokens == 0:
                        raw_tokens, prepared_tokens, prompt_tokens = 1, 1, 1
                    
                    df_tokens = token_analyzer.prepare_viz_data(raw_tokens, prepared_tokens, prompt_tokens)
                    ui_components.render_token_analysis(df_tokens)
                    ui_components.render_prompt_popovers(response.get("prompt_views", {}))
                    
                    # Execution flow visualization
                    ui_components.render_execution_timeline(pad.load(), query)

    with col2:
        st.subheader("System Status")
        st.metric("Vector Store", "Active", delta="Ready")
        st.metric("Graph DB", "Connected" if graph_retriever.driver else "Disconnected", delta_color="normal")
        
        # Neo4j Access
        st.markdown("### Neo4j Access")
        st.link_button("Open Neo4j Browser", "http://localhost:7474")
        st.caption(f"**User:** `{config.NEO4J_USER}`")
        st.caption(f"**Password:** `{config.NEO4J_PASSWORD}`")
        
        ui_components.render_scratchpad(pad.load())
            
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("Clear Scratchpad"):
                pad.clear()
                st.rerun()
        with col_s2:
            with st.popover("View Raw JSON"):
                st.json(pad.load())

        st.markdown("---")
        if st.button("Reset All Stores", type="primary"):
            if st.checkbox("Confirm Reset?"):
                doc_processor.reset_stores()
                # Add graph reset here if needed
                st.error("Stores reset!")

if __name__ == "__main__":
    main()