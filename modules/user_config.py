"""
User Configuration Module
Handles user input for environment variables through Streamlit UI
"""
import streamlit as st
import os
from typing import Dict, Optional

# List of environment variables that can be configured
ENV_VARIABLES = {
    "OPENAI_API_KEY": {
        "label": "OpenAI API Key",
        "type": "password",
        "required": True,
        "help": "Your OpenAI API key (starts with 'sk-')"
    },
    "NEO4J_URI": {
        "label": "Neo4j URI",
        "type": "text",
        "required": False,
        "default": "bolt://localhost:7687",
        "help": "Neo4j database connection URI"
    },
    "NEO4J_USER": {
        "label": "Neo4j Username",
        "type": "text",
        "required": False,
        "default": "neo4j",
        "help": "Neo4j database username"
    },
    "NEO4J_PASSWORD": {
        "label": "Neo4j Password",
        "type": "password",
        "required": False,
        "default": "password",
        "help": "Neo4j database password"
    }
}


def get_user_config() -> Dict[str, Optional[str]]:
    """
    Get configuration from user input (session state) or return None if not set.
    Returns a dictionary of environment variable names and their values.
    """
    if 'user_config' not in st.session_state:
        return {}
    return st.session_state.user_config


def render_config_form() -> Dict[str, Optional[str]]:
    """
    Render input form for all environment variables.
    Returns dictionary of configured values.
    """
    # Initialize session state if not exists
    if 'user_config' not in st.session_state:
        st.session_state.user_config = {}
    
    # Load existing .env values as defaults if available
    existing_values = {}
    for key in ENV_VARIABLES.keys():
        env_value = os.getenv(key)
        if env_value:
            existing_values[key] = env_value
    
    config_values = {}
    
    with st.expander("🔧 Configuration Settings", expanded=not st.session_state.user_config):
        st.markdown("Enter your configuration details below. These will override .env file values.")
        
        with st.form("config_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            for idx, (key, config) in enumerate(ENV_VARIABLES.items()):
                # Get default value (from session state, then .env, then config default)
                default_val = (
                    st.session_state.user_config.get(key) or 
                    existing_values.get(key) or 
                    config.get("default", "")
                )
                
                # Alternate between columns for better layout
                col = col1 if idx % 2 == 0 else col2
                
                with col:
                    if config["type"] == "password":
                        value = st.text_input(
                            config["label"],
                            value=default_val,
                            type="password",
                            help=config["help"],
                            key=f"input_{key}"
                        )
                    else:
                        value = st.text_input(
                            config["label"],
                            value=default_val,
                            help=config["help"],
                            key=f"input_{key}"
                        )
                    
                    config_values[key] = value if value else None
                    
                    # Show required indicator
                    if config["required"] and not value:
                        st.caption(f"⚠️ Required")
            
            submitted = st.form_submit_button("💾 Save Configuration", type="primary", use_container_width=True)
            
            if submitted:
                # Validate required fields
                missing_fields = []
                for key, config in ENV_VARIABLES.items():
                    if config["required"] and not config_values[key]:
                        missing_fields.append(config["label"])
                
                if missing_fields:
                    st.error(f"Please fill in required fields: {', '.join(missing_fields)}")
                else:
                    # Save to session state
                    st.session_state.user_config = config_values
                    st.success("✅ Configuration saved successfully!")
                    st.rerun()
    
    # Clear button outside form
    if st.session_state.user_config:
        if st.button("🗑️ Clear Configuration", use_container_width=True, key="clear_config_btn"):
            clear_config()
            st.rerun()
    
    return st.session_state.user_config


def render_config_display():
    """
    Display current configuration values (masked for sensitive data).
    """
    config = get_user_config()
    
    if not config:
        st.info("No configuration set. Please fill in the form above.")
        return
    
    st.markdown("#### Current Configuration")
    with st.expander("View Configuration", expanded=False):
        for key, config_item in ENV_VARIABLES.items():
            value = config.get(key)
            if value:
                # Mask sensitive values
                if config_item["type"] == "password":
                    masked = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "*" * len(value)
                    st.text(f"{config_item['label']}: {masked}")
                else:
                    st.text(f"{config_item['label']}: {value}")
            else:
                st.text(f"{config_item['label']}: Not set")


def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a configuration value from user input (session state), 
    falling back to environment variable or default.
    
    Args:
        key: Environment variable name
        default: Default value if not found
    
    Returns:
        Configuration value or default
    """
    # Priority: user config > .env > default
    user_config = get_user_config()
    if key in user_config and user_config[key]:
        return user_config[key]
    
    env_value = os.getenv(key)
    if env_value:
        return env_value
    
    return default


def clear_config():
    """Clear user configuration from session state."""
    if 'user_config' in st.session_state:
        st.session_state.user_config = {}

