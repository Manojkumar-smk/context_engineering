import streamlit as st
import altair as alt
import pandas as pd

import config

def render_header():
    st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON, layout="wide")
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.markdown("---")

def render_sidebar(retrieval_mode_enum, role_prompts_class):
    with st.sidebar:
        st.header("Configuration")
        st.info("Configure your RAG pipeline settings here.")
        
        st.markdown("### 1. Ingestion Settings")
        ingestion_mode = st.radio(
            "Ingestion Mode",
            ["Traditional RAG (Vector Store)", "Knowledge Graph", "Both"],
            index=2,
            help="Choose how to process and store the documents."
        )
        
        st.markdown("---")
        st.markdown("### 2. Retrieval Settings")
        mode_options = [m.value for m in retrieval_mode_enum]
        selected_mode = st.selectbox(
            "Retrieval Mode", 
            mode_options,
            help="Choose how the system retrieves information."
        )
        
        # Tooltip for mode
        mode_enum_val = next(m for m in retrieval_mode_enum if m.value == selected_mode)
        # We need to import retrieval_modes here or pass the function, but simpler to just pass the enum and do logic in app or here.
        # To avoid circular imports if we import retrieval_modes here, we'll just return the string.
        
        st.markdown("---")
        st.markdown("### 3. Agent Settings")
        role_options = ["Normal Chatbot", "Coding Agent", "Document Analyser"]
        selected_role = st.selectbox("Agent Role", role_options)
        
        # Show role prompt preview
        st.caption(role_prompts_class.get_prompt(selected_role).strip().split("\n")[0])
        
        return ingestion_mode, selected_mode, selected_role
        
def render_file_upload_stats(files, chunks):
    if files and chunks:
        st.success(f"Processed {len(files)} files into {len(chunks)} chunks.")
        st.metric("Total Chunks", len(chunks))
        
def render_structured_answer(response: dict):
    confidence = response.get("confidence", "Low")
    color = "green" if confidence == "High" else "orange" if confidence == "Medium" else "red"
    
    st.markdown(f"### Answer (Confidence: :{color}[{confidence}])")
    st.write(response.get("answer"))
    
    with st.expander("Sources & Evidence"):
        tools = response.get("tools_used", [])
        if tools:
            st.markdown("**Tools Utilized:**")
            for tool in tools:
                st.markdown(f"- {tool}")
            st.markdown("---")
        for source in response.get("sources", []):
            st.markdown(f"- `{source}`")
    
    if response.get("limitations"):
        st.warning(f"**Limitations:** {response.get('limitations')}")

def render_token_analysis(df: pd.DataFrame):
    st.subheader("Token Usage Analysis")
    st.bar_chart(df, x="Stage", y="Tokens", color="Stage")
    
    # Calculate savings
    raw = df.loc[df["Stage"] == "Raw Context", "Tokens"].values[0]
    final = df.loc[df["Stage"] == "Final Prompt", "Tokens"].values[0]
    if raw > 0:
        savings = ((raw - final) / raw) * 100
        st.metric("Token Reduction", f"{savings:.1f}%", help="Reduction from raw retrieval to final prompt.")

def render_scratchpad(history: list):
    st.subheader("Scratchpad History")
    filter_text = st.text_input("Filter logs", placeholder="Search query or step...")
    
    for item in reversed(history):
        if filter_text.lower() in str(item).lower():
            # Handle different item structures
            timestamp = item.get("timestamp", "No Timestamp")
            step = item.get("step", "Log")
            query = item.get("query", "")
            content = item.get("content", str(item))
            
            label = f"{timestamp} - {step}"
            if query:
                label += f" ({query[:30]}...)"
            
            with st.expander(label):
                st.write(content)
                if "metadata" in item:
                    st.json(item["metadata"])
                elif isinstance(item, dict):
                     st.json(item)

def render_prompt_popovers(prompt_views: dict):
    if not prompt_views:
        return
    st.markdown("### Prompt Inspector")
    labels = [
        ("Raw Context", "raw_context"),
        ("Prepared Context", "prepared_context"),
        ("Final Prompt", "final_prompt"),
    ]
    cols = st.columns(len(labels))
    for col, (label, key) in zip(cols, labels):
        content = prompt_views.get(key)
        with col:
            if content:
                with st.popover(label):
                    st.write(content)
            else:
                st.caption(f"{label} unavailable")

def render_execution_timeline(history: list, current_query: str):
    st.subheader("Execution Flow")
    if not history:
        st.info("No execution history yet.")
        return

    filtered = [entry for entry in history if entry.get("query") == current_query]
    if not filtered:
        st.info("Run a query to see its execution trace.")
        return

    ordered_entries = sorted(filtered, key=lambda e: e.get("timestamp", ""))

    timeline_df = pd.DataFrame(
        [
            {
                "order": idx,
                "step": entry.get("step", "Unknown"),
                "content": entry.get("content", ""),
                "timestamp": pd.to_datetime(entry.get("timestamp")),
            }
            for idx, entry in enumerate(ordered_entries, start=1)
        ]
    )

    timeline_df.sort_values("timestamp", inplace=True)
    timeline_df["label"] = timeline_df["step"] + " (" + timeline_df["timestamp"].dt.strftime("%H:%M:%S") + ")"

    chart = (
        alt.Chart(timeline_df)
        .mark_line(point=alt.OverlayMarkDef(size=90, filled=True))
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("order:O", title=None, axis=alt.Axis(labels=False, ticks=False)),
            color=alt.Color("step:N", legend=alt.Legend(title="Step")),
            tooltip=["step", "content", alt.Tooltip("timestamp:T", title="Time")],
        )
        .properties(height=180)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("#### Step Details")
    for entry in ordered_entries:
        ts = entry.get("timestamp")
        ts_display = ""
        try:
            ts_display = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_display = str(ts)
        st.markdown(f"**{entry.get('step', 'Log')}** · {ts_display}")
        st.caption(entry.get("content", ""))
        metadata = entry.get("metadata")
        if metadata:
            with st.expander("See metadata", expanded=False):
                st.json(metadata)
