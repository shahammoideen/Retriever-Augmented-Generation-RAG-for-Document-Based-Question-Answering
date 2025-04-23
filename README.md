# Retriever-Augmented-Generation-RAG-for-Document-Based-Question-Answering
This project is focused on implementing a Retriever-Augmented Generation (RAG) pipeline using LangChain, Hugging Face, and Sentence Transformers. The goal is to retrieve relevant information from a PDF document and generate answers to user queries based on its content.

Key Features:
PDF Document Loading: Uses PyPDFLoader from LangChain to load and process PDF documents.

Text Splitting: The document is split into smaller chunks using RecursiveCharacterTextSplitter to facilitate efficient retrieval and processing.

Sentence Embeddings: The text chunks are converted into embeddings using the SentenceTransformerEmbeddings model.

Vector Database: Chunks are stored in a Chroma vector database for efficient retrieval.

Retriever and Generator: A custom LLM class connects to the Hugging Face Inference API (using FLAN-T5) to generate answers based on the retrieved chunks.

Question Answering: Users can query the system to get answers like:

"What is Scaled Dot-Product Attention?"

"What is Multi-Head Attention?"
