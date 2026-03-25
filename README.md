# Resume RAG Q&A System

A modular Retrieval-Augmented Generation (RAG) system designed to interactively answer questions based on a structured domain dataset (a professional resume).

This project strictly separates data ingestion, vector search, and LLM generation into isolated, testable classes. It prioritizes privacy and speed by generating semantic embeddings locally while leveraging the OpenRouter API for high-quality, context-aware text generation.

## 🏗 Architecture

* **Data Source:** JSON-formatted resume/domain data (`data/resume.json`).
* **Embeddings:** Local, CPU-friendly HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`).
* **Vector Database:** In-memory `FAISS` index for exact L2 distance search.
* **LLM Generation:** OpenRouter API (Default: `stepfun/step-3.5-flash:free`).
* **Testing:** Automated unit and integration testing via `pytest` with API mocking.

## 📂 Repository Structure

* **data/**
* `resume.json` — The structured domain knowledge


* **src/**
* `__init__.py`
* `rag_qa.py` — Core logic (Ingestion, Indexing, Pipeline, CLI)
* `rag_qa.ipynb` — Interactive Jupyter Notebook interface


* **tests/**
* `__init__.py`
* `test_rag_qa.py` — Isolated unit tests with mocked API calls


* **Root Files**
* `.env` — Environment variables (API Keys - DO NOT COMMIT)
* `progress.md` — Project state and tracking
* `requirements.txt` — Python dependencies



## ⚙️ Prerequisites & Setup

**1. Clone the repository and navigate to the root directory.**

**2. Create and activate a virtual environment:**

* **Windows:** `python -m venv .venv` and then `.venv\Scripts\activate`
* **macOS/Linux:** `python -m venv .venv` and then `source .venv/bin/activate`

**3. Install the dependencies:**
Run `pip install -r requirements.txt`

**4. Configure Environment Variables:**
Create a `.env` file in the root directory and add your OpenRouter API key: `OPENROUTER_API_KEY=your_actual_api_key_here`

## 🚀 Usage

You can interact with the RAG pipeline using either the command-line interface or the Jupyter Notebook.

### Option A: Command Line Interface (CLI)

Run the core script from the root directory to start the interactive terminal loop:
`python src/rag_qa.py`

*Type your questions at the prompt, and type `exit` or `quit` to gracefully shut down the system.*

### Option B: Jupyter Notebook

Launch Jupyter and open `src/rag_qa.ipynb`.
`jupyter notebook`

* **Note for IDE Users:** If your IDE requests a token to run the notebook server, open your terminal, ensure your virtual environment is active, and run `jupyter server list`. Copy the token string provided in the output to authenticate your session.
* Execute the setup cell, then use the query cell to iteratively test different questions.

## 🧪 Testing

This project uses `pytest` for Test-Driven Development. The test suite includes fixtures that dynamically generate mock JSON data and intercept HTTP requests to ensure tests are fast, offline, and reliable.

To run the test suite from the root directory, execute:
`pytest -v tests/`
