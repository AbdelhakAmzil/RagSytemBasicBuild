import unittest
import os
from unittest.mock import patch, MagicMock
from rag_system import DocumentProcessor, VectorStore, RAGSystem

class TestDocumentProcessor(unittest.TestCase):
    """Tests for DocumentProcessor"""

    def setUp(self):
        """Set up test environment"""
        self.processor = DocumentProcessor()

    @patch("rag_system.PdfReader")
    def test_process_pdf(self, mock_pdf_reader):
        """Test PDF processing"""
        mock_pdf_reader.return_value.pages = [MagicMock(extract_text=lambda: "This is a test PDF page.")]
        result = self.processor.process_pdf("test.pdf")
        self.assertEqual(result, ["This is a test PDF page."])

    @patch("rag_system.docx.Document")
    def test_process_docx(self, mock_doc):
        """Test DOCX processing"""
        mock_doc.return_value.paragraphs = [MagicMock(text="Test paragraph 1"), MagicMock(text="Test paragraph 2")]
        result = self.processor.process_docx("test.docx")
        self.assertEqual([text.strip() for text in result], ["Test paragraph 1\nTest paragraph 2"])

    def test_process_txt(self):
        """Test TXT processing"""
        with open("test.txt", "w", encoding="utf-8") as f:
            f.write("Test line 1\n\nTest line 2")
        result = self.processor.process_txt("test.txt")
        os.remove("test.txt")  # Clean up after test
        self.assertEqual(result, ["Test line 1", "Test line 2"])

    def test_process_document(self):
        """Test document processing based on file extension"""
        with patch.object(self.processor, "process_pdf", return_value=["PDF text"]) as mock_pdf:
            result = self.processor.process_document("test.pdf")
            mock_pdf.assert_called_once()
            self.assertEqual(result, ["PDF text"])

class TestVectorStore(unittest.TestCase):
    """Tests for VectorStore"""

    @patch("rag_system.SentenceTransformer")
    @patch("rag_system.chromadb.Client")
    def setUp(self, mock_chromadb, mock_embedder):
        """Set up test environment"""
        self.mock_embedder = mock_embedder.return_value
        self.mock_embedder.encode.return_value = [[0.1, 0.2, 0.3]]
        self.mock_chromadb = mock_chromadb.return_value
        self.mock_collection = MagicMock()
        self.mock_chromadb.get_collection.return_value = self.mock_collection
        self.store = VectorStore()

    def test_add_documents(self):
        """Test adding documents to vector store"""
        self.store.add_documents(["Test document"], [{"source": "test.txt"}])
        self.mock_collection.add.assert_called_once()

    def test_search(self):
        """Test searching in vector store"""
        self.mock_collection.query.return_value = {
            "documents": [["Relevant document"]],
            "metadatas": [[{"source": "test.txt"}]]
        }
        results = self.store.search("query")
        self.assertEqual(results, [{"content": "Relevant document", "metadata": {"source": "test.txt"}}])

class TestRAGSystem(unittest.TestCase):
    """Tests for RAGSystem"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def setUp(self, mock_processor, mock_vector_store):
        """Set up test environment"""
        self.mock_processor = mock_processor.return_value
        self.mock_processor.process_document.return_value = ["Processed text"]
        self.mock_vector_store = mock_vector_store.return_value
        self.mock_vector_store.add_documents.return_value = None
        self.mock_vector_store.search.return_value = [{"content": "Relevant doc", "metadata": {"source": "test.pdf"}}]
        self.rag_system = RAGSystem(api_key="fake-key")

    def test_add_document(self):
        """Test adding a document to RAG system"""
        count = self.rag_system.add_document("test.pdf")
        self.mock_processor.process_document.assert_called_once_with("test.pdf")
        self.mock_vector_store.add_documents.assert_called_once()
        self.assertEqual(count, 1)

    @patch("rag_system.openai.chat.completions.create")
    def test_generate_response(self, mock_openai):
        """Test response generation"""
        mock_openai.return_value.choices = [MagicMock(message=MagicMock(content="Generated response"))]
        response = self.rag_system.generate_response("What is AI?")
        self.assertIn("Generated response", response)

if __name__ == "__main__":
    unittest.main()
