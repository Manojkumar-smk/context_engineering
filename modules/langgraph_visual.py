import streamlit as st

class LangGraphVisualizer:
    """
    Visualizes the agent workflow (LangGraph).
    """
    
    @staticmethod
    def render_graph(current_step: str = None):
        """
        Renders a static graphviz chart of the RAG workflow, highlighting the current step.
        """
        # Define the graph structure
        # Nodes: Ingest -> Retrieve -> Select -> Compress -> Evaluate -> Answer
        # Edges: Flow
        
        graph_attr = {
            "rankdir": "LR",
            "bgcolor": "transparent"
        }
        
        node_attr = {
            "shape": "box",
            "style": "filled",
            "fillcolor": "#f0f2f6",
            "color": "#31333F",
            "fontname": "Sans-Serif"
        }
        
        edge_attr = {
            "color": "#31333F"
        }
        
        # Highlight color
        active_fill = "#ff4b4b" # Streamlit red
        active_font = "white"
        
        dot = "digraph RAGWorkflow {\n"
        dot += f'  rankdir="{graph_attr["rankdir"]}";\n'
        dot += f'  bgcolor="{graph_attr["bgcolor"]}";\n'
        
        steps = ["Ingest", "Retrieve", "Select", "Compress", "Evaluate", "Answer"]
        
        for step in steps:
            fill = active_fill if step == current_step else node_attr["fillcolor"]
            font = active_font if step == current_step else "black"
            dot += f'  {step} [label="{step}", shape="{node_attr["shape"]}", style="{node_attr["style"]}", fillcolor="{fill}", fontcolor="{font}", fontname="{node_attr["fontname"]}"];\n'
            
        # Edges
        dot += "  Ingest -> Retrieve;\n"
        dot += "  Retrieve -> Select;\n"
        dot += "  Select -> Compress;\n"
        dot += "  Compress -> Evaluate;\n"
        dot += "  Evaluate -> Answer [label=\"Good Quality\"];\n"
        dot += "  Evaluate -> Retrieve [label=\"Poor Quality (Correction)\", style=\"dashed\"];\n"
        
        dot += "}"
        
        st.graphviz_chart(dot)
