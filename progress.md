# Project Progress: RAG-Powered Q&A System

## Overview
A modular Python project implementing a Retrieval-Augmented Generation system using a local embedding model, an in-memory FAISS vector database, and the OpenRouter API for generative responses.

## Current State: Phase 1
- [x] Define architecture and repository scope.
- [x] Generate structured domain data (`data/resume.json`).
- [x] Initialize `progress.md` and repository structure via Bash script.
- [x] Implement `DocumentProcessor` in `src/rag_qa.py` (Ingestion, Chunking, FAISS Indexing).
- [x] Implement isolated unit tests in `tests/test_rag_qa.py`.
- [x] Implement OpenRouter LLM generation class.
- [x] Connect Retrieval pipeline to Generation pipeline (QA Chain).
- [x] Finalize interactive CLI / Notebook interface.
