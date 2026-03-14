"""
rag_qa.py
A highly modular Retrieval-Augmented Generation (RAG) implementation.
Phase 1: Data Ingestion, Semantic Chunking, and FAISS Vectorization.
"""

import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import os
import requests
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Configure logging for robust error tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DocumentProcessor:
    """
    Handles loading JSON data, converting it to semantic text chunks,
    generating vector embeddings, and creating a searchable FAISS index.
    """

    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the processor with a lightweight local embedding model.

        Parameters:
            embedding_model (str): HuggingFace model identifier. Defaults to a CPU-friendly model.
        """
        try:
            logging.info(f"Loading embedding model: {embedding_model}...")
            # Load the model locally. This will download on the first run, then cache.
            self.model = SentenceTransformer(embedding_model)
            # Store the embedding dimension required for initializing FAISS
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logging.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logging.error(f"Failed to load embedding model. Error: {e}")
            raise

    def process_resume_json(self, filepath: str) -> List[str]:
        """
        Loads the structured JSON and flattens it into semantic text chunks.
        Semantic chunking preserves context better than arbitrary character splitting.

        Parameters:
            filepath (str): Path to the JSON data file.

        Returns:
            List[str]: A list of coherent text chunks ready for embedding.
        """
        chunks = []
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # 1. Chunk Personal Information
            if 'personal_information' in data:
                info = data['personal_information']
                chunks.append(
                    f"Profile: {info.get('name', 'N/A')} is located in {info.get('location', 'N/A')}. "
                    f"Contact via email at {info.get('email', 'N/A')} or phone at {info.get('phone', 'N/A')}. "
                    f"Summary: {info.get('summary', 'N/A')}"
                )

            # 2. Chunk Education History
            for edu in data.get('education', []):
                chunks.append(
                    f"Education: Studied {edu.get('degree')} at {edu.get('institution')} "
                    f"from {edu.get('timeline')}."
                )

            # 3. Chunk Professional Experience
            for exp in data.get('experience', []):
                responsibilities = " ".join(exp.get('responsibilities', []))
                chunks.append(
                    f"Experience: Worked as {exp.get('role')} at {exp.get('company')} "
                    f"({exp.get('timeline')}). Responsibilities included: {responsibilities}"
                )

            # 4. Chunk Projects
            for proj in data.get('projects', []):
                tech_stack = ", ".join(proj.get('technologies', []))
                chunks.append(
                    f"Project: Built {proj.get('name')} using {tech_stack}. "
                    f"Description: {proj.get('description')}"
                )

            # 5. Chunk Skills
            if 'skills' in data:
                skills = data['skills']
                chunks.append(f"Technical Skills - Languages: {', '.join(skills.get('languages', []))}")
                chunks.append(f"Technical Skills - Tools & Frameworks: {', '.join(skills.get('tools_and_frameworks', []))}")
                chunks.append(f"Core Competencies: {', '.join(skills.get('core_competencies', []))}")

            logging.info(f"Successfully extracted {len(chunks)} semantic chunks from {filepath}.")
            return chunks

        except FileNotFoundError:
            logging.error(f"Data file not found at {filepath}.")
            raise
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON in {filepath}. Ensure it is formatted correctly.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during JSON processing: {e}")
            raise

    def create_vector_store(self, chunks: List[str]) -> Tuple[faiss.IndexFlatL2, np.ndarray, List[str]]:
        """
        Generates embeddings for the text chunks and builds an in-memory FAISS index.

        Parameters:
            chunks (List[str]): The semantic text chunks to embed.

        Returns:
            Tuple containing the FAISS index, the raw embeddings array, and the original chunks.
        """
        try:
            logging.info("Generating embeddings for text chunks...")
            # Convert list of strings into a dense numpy array of float32 embeddings
            embeddings = self.model.encode(chunks, convert_to_numpy=True)

            logging.info("Initializing FAISS vector index...")
            # Use IndexFlatL2 for exact distance search (L2 distance/Euclidean)
            index = faiss.IndexFlatL2(self.embedding_dim)

            # Populate the index with our computed embeddings
            index.add(embeddings)
            logging.info(f"Successfully added {index.ntotal} vectors to the FAISS index.")

            return index, embeddings, chunks

        except Exception as e:
            logging.error(f"Failed to create vector store: {e}")
            raise

class OpenRouterGenerator:
    """
    Handles communication with the OpenRouter API to generate context-aware answers.
    """

    def __init__(self, model: str = "stepfun/step-3.5-flash:free"):
        """
        Initializes the generator with the specified OpenRouter model.

        Parameters:
            model (str): The model identifier on OpenRouter.
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logging.warning("OPENROUTER_API_KEY not found in environment variables.")

        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/tobekami",  # OpenRouter requires a referer
            "X-Title": "Resume RAG QA",
            "Content-Type": "application/json"
        }

    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Constructs the prompt with retrieved context and calls the OpenRouter API.

        Parameters:
            query (str): The user's question.
            context (List[str]): The semantic chunks retrieved from FAISS.

        Returns:
            str: The generated text response.
        """
        # Join the retrieved chunks into a single context block
        context_text = "\n".join(context)

        # Construct a strict system prompt to prevent hallucination
        system_prompt = (
            "You are an AI assistant answering questions about a specific individual based strictly on the provided resume context. "
            "If the answer is not contained in the context, say 'I do not have enough information to answer that.' "
            "Do not invent or hallucinate information."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2  # Low temperature for factual consistency
        }

        try:
            logging.info(f"Sending query to OpenRouter using model: {self.model}...")
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes

            data = response.json()
            return data['choices'][0]['message']['content'].strip()

        except requests.exceptions.RequestException as e:
            logging.error(f"OpenRouter API request failed: {e}")
            return "Error: Could not connect to the generation API."
        except KeyError as e:
            logging.error(f"Unexpected API response format: {e}")
            return "Error: Received malformed response from OpenRouter."


class RAGPipeline:
    """
    Orchestrates the Retrieval-Augmented Generation flow by connecting
    the DocumentProcessor (retriever) and the OpenRouterGenerator (LLM).
    """

    def __init__(self, processor: DocumentProcessor, generator: OpenRouterGenerator):
        self.processor = processor
        self.generator = generator
        self.index = None
        self.stored_chunks = []

    def setup(self, data_filepath: str):
        """Processes the JSON and builds the FAISS index."""
        chunks = self.processor.process_resume_json(data_filepath)
        self.index, _, self.stored_chunks = self.processor.create_vector_store(chunks)
        logging.info("RAG Pipeline setup complete.")

    def query(self, user_question: str, top_k: int = 3) -> str:
        """
        Executes the full RAG pipeline: embeds the query, retrieves the top_k chunks,
        and generates an answer.
        """
        if self.index is None:
            raise ValueError("Pipeline not initialized. Call setup() first.")

        try:
            logging.info(f"Processing query: '{user_question}'")
            # 1. Embed the user's question
            query_embedding = self.processor.model.encode([user_question], convert_to_numpy=True)

            # 2. Retrieve the top_k most similar chunks from FAISS
            distances, indices = self.index.search(query_embedding, top_k)

            # 3. Extract the actual text chunks using the retrieved indices
            retrieved_context = [self.stored_chunks[i] for i in indices[0] if i != -1]

            logging.info(f"Retrieved {len(retrieved_context)} relevant context chunks.")

            # 4. Pass the context and question to the LLM
            answer = self.generator.generate_answer(user_question, retrieved_context)
            return answer

        except Exception as e:
            logging.error(f"Failed to execute RAG query: {e}")
            raise


def main():
    """
    Interactive CLI loop for querying the RAG pipeline.
    Initializes the system, indexes the data, and awaits user questions.
    """
    print("\n" + "=" * 50)
    print("🚀 Resume RAG QA System Initializing...")
    print("=" * 50)

    # Define the path to your structured data
    data_filepath = '../data/resume.json' if not os.path.exists('data/resume.json') else 'data/resume.json'

    # Ensure the data file exists before spinning up the models
    if not os.path.exists(data_filepath):
        logging.error(
            f"Data file not found at {data_filepath}. Please ensure your resume.json is in the data/ directory.")
        return

    try:
        # Initialize the core RAG components
        processor = DocumentProcessor()
        generator = OpenRouterGenerator()

        # Setup and populate the pipeline
        pipeline = RAGPipeline(processor, generator)
        print("\n📚 Indexing resume data into FAISS...")
        pipeline.setup(data_filepath)

        print("\n✅ System ready! Type 'exit' or 'quit' to stop.")
        print("-" * 50)

        # Start the interactive query loop
        while True:
            user_question = input("\n🤔 Ask a question about the resume: ").strip()

            # Handle exit commands
            if user_question.lower() in ['exit', 'quit']:
                print("Exiting the QA system. Goodbye!")
                break

            # Skip empty inputs
            if not user_question:
                continue

            print("⏳ Retrieving context and generating answer...")

            # Execute the RAG query
            answer = pipeline.query(user_question, top_k=3)

            # Display the result
            print("\n🤖 Answer:")
            print(answer)
            print("-" * 50)

    except Exception as e:
        logging.error(f"A critical error occurred during execution: {e}")


if __name__ == "__main__":
    main()