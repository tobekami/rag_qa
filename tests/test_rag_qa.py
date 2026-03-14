"""
test_rag_qa.py
Automated unit tests for the RAG data ingestion and vectorization modules.
"""

import json
import pytest
from src.rag_qa import DocumentProcessor


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def dummy_resume_json(tmp_path):
    """
    Creates a temporary JSON file with mock resume data to isolate
    tests from the production data directory.
    """
    data = {
        "personal_information": {
            "name": "Jane Doe",
            "location": "Test City",
            "email": "jane@example.com",
            "phone": "555-0000",
            "summary": "A mock software engineer."
        },
        "education": [
            {
                "institution": "Test University",
                "degree": "BSc Computer Science",
                "timeline": "2020 - 2024"
            }
        ]
    }
    # Write the mock data to a temporary file managed by pytest
    filepath = tmp_path / "dummy_resume.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    return str(filepath)


@pytest.fixture(scope="module")
def processor():
    """
    Initializes the DocumentProcessor once for the entire test module
    to avoid repeatedly loading the embedding model into memory.
    """
    return DocumentProcessor()


# ==========================================
# TEST CASES
# ==========================================

def test_initialization(processor):
    """Verifies that the embedding model loads correctly with the expected dimensions."""
    assert processor.embedding_dim == 384, "Embedding dimension mismatch for all-MiniLM-L6-v2."


def test_process_resume_json(processor, dummy_resume_json):
    """Ensures JSON is correctly parsed and flattened into semantic chunks."""
    chunks = processor.process_resume_json(dummy_resume_json)

    # We expect 2 chunks based on our dummy data (1 personal info, 1 education)
    assert len(chunks) == 2, f"Expected 2 chunks, but got {len(chunks)}."
    assert isinstance(chunks[0], str), "Chunks must be returned as strings."
    assert "Jane Doe" in chunks[0], "Expected data not found in chunk."


def test_create_vector_store(processor, dummy_resume_json):
    """Validates that FAISS indexing generates the correct number of vectors."""
    chunks = processor.process_resume_json(dummy_resume_json)
    index, embeddings, stored_chunks = processor.create_vector_store(chunks)

    assert index.ntotal == len(chunks), "FAISS index count does not match input chunk count."
    assert embeddings.shape[1] == processor.embedding_dim, "Embedding matrix dimension mismatch."
    assert len(stored_chunks) == len(chunks), "Stored chunks length mismatch."


from unittest.mock import patch, MagicMock
from src.rag_qa import OpenRouterGenerator, RAGPipeline


# ==========================================
# GENERATOR & PIPELINE TESTS
# ==========================================

@pytest.fixture
def mock_generator():
    """Initializes the OpenRouterGenerator with a dummy API key for testing."""
    with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'dummy_key'}):
        return OpenRouterGenerator()


@patch('src.rag_qa.requests.post')
def test_generate_answer_success(mock_post, mock_generator):
    """Verifies that the generator correctly parses a successful API response."""
    # Mock the OpenRouter API JSON response structure
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'Jane Doe is a software engineer.'}}]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    query = "Who is Jane Doe?"
    context = ["Profile: Jane Doe is located in Test City. Summary: A mock software engineer."]

    answer = mock_generator.generate_answer(query, context)

    assert "Jane Doe is a software engineer" in answer
    # Verify the API was called exactly once
    mock_post.assert_called_once()


def test_rag_pipeline_integration(processor, mock_generator, dummy_resume_json):
    """
    Tests the full pipeline flow: indexing data, retrieving context,
    and passing it to the mocked generator.
    """
    # 1. Setup the pipeline
    pipeline = RAGPipeline(processor, mock_generator)
    pipeline.setup(dummy_resume_json)

    # Verify setup populated the FAISS index
    assert pipeline.index is not None
    assert len(pipeline.stored_chunks) == 2

    # 2. Mock the generator's response to bypass the actual API call
    with patch.object(mock_generator, 'generate_answer', return_value="Mocked pipeline answer.") as mock_gen_method:
        answer = pipeline.query("What degree does Jane have?", top_k=1)

        assert answer == "Mocked pipeline answer."
        # Verify the generator was called with the retrieved context
        mock_gen_method.assert_called_once()

        # Check that context was actually passed to the generator
        called_args, called_kwargs = mock_gen_method.call_args
        passed_context = called_args[1]
        assert len(passed_context) == 1  # We requested top_k=1