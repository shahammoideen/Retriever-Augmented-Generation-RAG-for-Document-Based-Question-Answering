#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import InferenceClient
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from typing import Optional, List

# Load PDF Document
pdf_path = "/content/attention.pdf"
pdf = PyPDFLoader(pdf_path)
doc = pdf.load()

# Text Splitting to handle large documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(doc)

# Sentence Transformer Embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

# Creating the Vector Database (Chroma)
db = Chroma.from_documents(chunks, embeddings)

# Hugging Face Inference API
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your-huggingface-api-token"
client = InferenceClient(model="google/flan-t5-base", token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

# Custom LLM Class for Hugging Face API Integration
class HuggingFaceInferenceAPI(LLM):
    client: InferenceClient
    task: str
    temperature: float = 0.1
    max_length: int = 200

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference_api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.client.text_generation(prompt, temperature=self.temperature, max_new_tokens=self.max_length)
        return response

# Create the LLM instance
llm = HuggingFaceInferenceAPI(client=client, task="text2text-generation", model_kwargs={"temperature": 0.1, "max_length": 512})

# Create a retriever from the Chroma database
retriever = db.as_retriever()

# Define the QA Chain for retrieval-based question answering
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Test Queries and Responses
query1 = "What is Scaled Dot-Product Attention?"
response1 = qa_chain.invoke(query1)
print("\n🔹 Answer:", response1)

query2 = "What is Multi-Head Attention?"
response2 = qa_chain.invoke(query2)
print("\n🔹 Answer:", response2)


# In[ ]:




