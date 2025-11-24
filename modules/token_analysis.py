import tiktoken
import config
import pandas as pd

class TokenAnalyzer:
    """
    Analyzes token usage, estimates costs, and prepares visualization data.
    """
    
    def __init__(self, model_name: str = config.LLM_MODEL_NAME):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = config.LLM_MODEL_NAME) -> float:
        pricing = config.PRICING.get(model, {"input": 0, "output": 0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def prepare_viz_data(self, raw_tokens: int, prepared_tokens: int, prompt_tokens: int):
        """
        Prepares a DataFrame for Altair/Streamlit bar charts.
        """
        data = {
            "Stage": ["Raw Context", "Prepared Context", "Final Prompt"],
            "Tokens": [raw_tokens, prepared_tokens, prompt_tokens],
            "Cost ($)": [
                self.estimate_cost(raw_tokens, 0),
                self.estimate_cost(prepared_tokens, 0),
                self.estimate_cost(prompt_tokens, 0)
            ]
        }
        return pd.DataFrame(data)

    def get_educational_note(self, metric: str) -> str:
        notes = {
            "reduction": "Context selection and compression significantly reduce token usage, lowering costs and focusing the LLM on relevant info.",
            "cost": "Costs are estimated based on current OpenAI pricing per 1M tokens.",
            "tiktoken": "Token counts are exact calculations using the tiktoken library, matching OpenAI's tokenizer."
        }
        return notes.get(metric, "")
