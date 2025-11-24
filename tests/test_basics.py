import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.token_analysis import TokenAnalyzer
from modules.document_processing import DocumentProcessor
import config

def test_token_counting():
    analyzer = TokenAnalyzer()
    text = "Hello world"
    # "Hello world" is usually 2 tokens in cl100k_base
    count = analyzer.count_tokens(text)
    assert count > 0
    assert isinstance(count, int)

def test_cost_estimation():
    analyzer = TokenAnalyzer()
    cost = analyzer.estimate_cost(1000, 1000, "gpt-4o")
    assert cost > 0
    # 1000 input * 5/1M + 1000 output * 15/1M = 0.005 + 0.015 = 0.02
    assert abs(cost - 0.02) < 0.0001

def test_chunking_config():
    # Verify config values are reasonable
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE

def test_document_processor_initialization():
    processor = DocumentProcessor()
    assert processor.text_splitter is not None
    assert processor.embeddings is not None
