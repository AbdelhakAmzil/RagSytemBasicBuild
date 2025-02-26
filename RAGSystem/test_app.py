import unittest
from unittest.mock import patch, MagicMock
import os
from app import RAGSystem
from rag_system import DocumentProcessor, VectorStore

class TestDocumentProcessor(unittest.TestCase):
    
    def setUp(self):
        """Setup common resources before each test."""
        self.processor = DocumentProcessor()

    @patch("app.PdfReader")
    def test_process_pdf(self, MockPdfReader):
        """Test PDF processing with mock."""
        mock_pdf = MockPdfReader.return_value
        mock_pdf.pages = [MagicMock(extract_text=lambda: "Page 1 text\n\nPage 2 text")]
        
        result = self.processor.process_pdf("dummy.pdf")
        self.assertEqual(result, ["Page 1 text", "Page 2 text"])

    @patch("app.docx.Document")
    def test_process_docx(self, MockDocx):
        """Test DOCX processing with mock."""
        mock_doc = MockDocx.return_value
        mock_doc.paragraphs = [MagicMock(text="Paragraph 1"), MagicMock(text="Paragraph 2")]
        
        result = self.processor.process_docx("dummy.docx")
        self.assertEqual(result, ["Paragraph 1\nParagraph 2"])

    def test_process_txt(self):
        """Test TXT processing."""
        with patch("builtins.open", unittest.mock.mock_open(read_data="Line 1\n\nLine 2")):
            result = self.processor.process_txt("dummy.txt")
        self.assertEqual(result, ["Line 1", "Line 2"])

    def test_process_document_invalid(self):
        """Test processing an unsupported file type."""
        with self.assertRaises(ValueError):
            self.processor.process_document("unsupported.xyz")


class TestVectorStore(unittest.TestCase):

    @patch("app.SentenceTransformer")
    @patch("app.chromadb.Client")
    def setUp(self, MockClient, MockSentenceTransformer):
        """Setup VectorStore with mocks."""
        self.mock_client = MockClient.return_value
        self.mock_collection = self.mock_client.get_collection.return_value
        self.mock_embedding_model = MockSentenceTransformer.return_value
        self.mock_embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]

        self.store = VectorStore()

    def test_add_documents(self):
        """Test adding documents to the vector store."""
        self.store.add_documents(["Document 1", "Document 2"])
        self.mock_collection.add.assert_called_once()

    def test_search(self):
        """Test searching in the vector store."""
        self.mock_collection.query.return_value = {
            "documents": [["Doc 1 content", "Doc 2 content"]],
            "metadatas": [[{"source": "file1.txt"}, {"source": "file2.txt"}]]
        }
        results = self.store.search("test query")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["metadata"]["source"], "file1.txt")


class TestRAGSystem(unittest.TestCase):

    @patch("app.VectorStore")
    def setUp(self, MockVectorStore):
        """Setup RAGSystem with mocks."""
        self.mock_vector_store = MockVectorStore.return_value
        self.rag = RAGSystem(api_key="test-key")

    @patch.object(DocumentProcessor, "process_document", return_value=["Chunk 1", "Chunk 2"])
    def test_add_document(self, mock_process_document):
        """Test adding a document."""
        num_chunks = self.rag.add_document("dummy.txt")
        self.assertEqual(num_chunks, 2)
        self.mock_vector_store.add_documents.assert_called_once()

    @patch("app.openai.chat.completions.create")
    def test_generate_response(self, mock_openai):
        """Test generating a response with OpenAI mock."""
        self.mock_vect
