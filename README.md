# Ollama Python Project

This project uses various Python libraries for document processing, embedding generation, LLM integration, and text-to-speech.

## Dependencies

- **ollama**: The official Python client for interacting with the local Ollama service, which runs large language models (LLMs) locally.
- **chromadb**: A fast, embedded open-source vector database used for storing and querying vector embeddings of text.
- **pdfplumber**: A library designed to extract text, tables, and other data accurately from PDF files.
- **langchain** & related core/community packages (`langchain-core`, `langchain-ollama`, `langchain-community`, `langchain-text-splitters`): A powerful framework to build and orchestrate applications powered by language models (such as Retrieval-Augmented Generation or AI agents). The specific sub-packages handle connecting to Ollama, splitting text into manageable chunks, and general community integrations.
- **unstructured** (and `unstructured[all-docs]`): A robust data processing tool for extracting and formatting raw data from complex documents (PDFs, HTML, Word docs) into clean text that is ready for machine learning pipelines.
- **fastembed**: A fast, lightweight Python library for generating low-latency text embeddings.
- **sentence-transformers**: A popular framework from Hugging Face that easily computes sentence, text, and image embeddings for dense information retrieval.
- **elevenlabs**: The official Python client for the ElevenLabs API, used for creating realistic text-to-speech (voice generation).
