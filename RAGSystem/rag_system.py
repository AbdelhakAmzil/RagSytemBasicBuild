import os
from typing import List, Dict, Any
import numpy as np

from dotenv import load_dotenv

# Document processing imports
from PyPDF2 import PdfReader
import docx

# Embedding and vector DB
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# LLM for generation (using OpenAI as example)
import openai

class DocumentProcessor:
    """Processes different document types into text chunks"""
    
    def process_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF and split into chunks"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Simple chunking by paragraphs (can be improved)
        chunks = [p for p in text.split("\n\n") if p.strip()]
        return chunks
    
    def process_docx(self, file_path: str) -> List[str]:
        """Extract text from DOCX and split into chunks"""
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        # Simple chunking by paragraphs
        chunks = [p for p in text.split("\n\n") if p.strip()]
        return chunks
    
    def process_txt(self, file_path: str) -> List[str]:
        """Process plain text files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Simple chunking by paragraphs
        chunks = [p for p in text.split("\n\n") if p.strip()]
        return chunks
    
    def process_document(self, file_path: str) -> List[str]:
        """Process document based on file extension"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            return self.process_pdf(file_path)
        elif ext == '.docx':
            return self.process_docx(file_path)
        elif ext == '.txt':
            return self.process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


class VectorStore:
    """Manages document embeddings and retrieval"""
    
    def __init__(self, collection_name: str = "documents"):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(collection_name)
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the vector store"""
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Prepare document IDs and metadata
        ids = [f"doc_{i}_{os.urandom(4).hex()}" for i in range(len(documents))]
        if metadata is None:
            metadata = [{"source": "unknown"} for _ in documents]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings, #embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadata
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents based on query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(documents, metadatas)
        ]


class RAGSystem:
    """Main RAG system combining document processing, embedding, and generation"""
    
    def __init__(self, api_key: str = None):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        
        # Setup OpenAI (if API key is provided)
        # if api_key:
        #     openai.api_key = api_key
        load_dotenv(override=True)
        
        # Prioriser la clé API fournie en paramètre
        api_key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")

        # Vérification si la clé API est bien récupérée
        if not api_key:
            raise ValueError("Aucune clé API fournie. Définissez-la en paramètre ou dans le fichier .env.")
    
    def add_document(self, file_path: str, metadata: Dict[str, Any] = None):
        """Process and add a document to the system"""
        # Process document into chunks
        chunks = self.document_processor.process_document(file_path)
        
        # Prepare metadata for each chunk
        if metadata is None:
            metadata = {"source": os.path.basename(file_path)}
        else:
            metadata["source"] = os.path.basename(file_path)
        
        chunk_metadata = [metadata.copy() for _ in chunks]
        
        # Add to vector store
        self.vector_store.add_documents(chunks, chunk_metadata)
        
        return len(chunks)
    
    def generate_response(self, query: str) -> str:
        """Generate a response to the user query"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(query)
        
        # Format context for the LLM
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        sources = [doc["metadata"]["source"] for doc in relevant_docs]
        unique_sources = list(set(sources))
        
        # Create prompt with context
        prompt = f"""
        Answer the following question based on the provided context.
        If you cannot answer based on the context, say "I don't have enough information to answer this question."
        
        Context:
        {context}
        
        Question: {query}
        
        Answer in French language as it's the user's preferred language.
        """
        
        # Generate response using OpenAI
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content
        
        # Add sources if any were found
        if unique_sources:
            sources_text = "\n\nSources: " + ", ".join(unique_sources)
            answer += sources_text
        
        return answer