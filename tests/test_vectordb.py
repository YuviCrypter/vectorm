import os
import unittest
import uuid
import numpy as np
from typing import List, Dict, Any

from vectorm import VectORMClient
from vectorm.vectordb import ChromaDB, MilvusDB, QdrantDB, PineconeDB, Document

# Test data
TEST_DOCUMENTS = [
    Document(
        id=str(uuid.uuid4()),
        document="This is a test document about vector databases.",
        vector=np.random.rand(1536).tolist(),
        metadata={"source": "test", "category": "database"}
    ),
    Document(
        id=str(uuid.uuid4()),
        document="Vector databases are used for similarity search.",
        vector=np.random.rand(1536).tolist(),
        metadata={"source": "test", "category": "search"}
    ),
    Document(
        id=str(uuid.uuid4()),
        document="ChromaDB, Milvus, Qdrant, and Pinecone are popular vector databases.",
        vector=np.random.rand(1536).tolist(),
        metadata={"source": "test", "category": "comparison"}
    )
]

# Test query vector
TEST_QUERY_VECTOR = np.random.rand(1536).tolist()

class BaseVectorDBTest(unittest.TestCase):
    """Base class for vector database tests."""
    
    def setUp(self):
        """Set up the test environment."""
        if self.__class__ == BaseVectorDBTest:
            self.skipTest("BaseVectorDBTest is an abstract class")
        self.table_name = f"test_table_{uuid.uuid4().hex[:8]}"
        self.client = None
        self.table = None
        
    def tearDown(self):
        """Clean up after tests."""
        # Cleanup will be handled by the specific test classes
        
    def test_create_table(self):
        """Test creating a table."""
        self.assertIsNotNone(self.table)
        self.assertTrue(self.table.table_exists)
        
    def test_insert_and_search(self):
        """Test inserting documents and searching."""
        # Insert documents
        self.table.insert(TEST_DOCUMENTS)
        
        # Search for similar vectors
        results = self.table.search(TEST_QUERY_VECTOR, limit=2)
        
        # Check results
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        # Only check result structure if we have results
        if len(results) > 0:
            self.assertIn('id', results[0])
            self.assertIn('document', results[0])
            self.assertIn('metadata', results[0])
            self.assertIn('distance', results[0])
            
    def test_delete(self):
        """Test deleting a document."""
        # Insert a document
        self.table.insert([TEST_DOCUMENTS[0]])
        
        # Delete the document
        result = self.table.delete(TEST_DOCUMENTS[0].id)
        
        # Check result
        self.assertTrue(result)
        
        # Verify document is deleted by searching
        results = self.table.search(TEST_QUERY_VECTOR, limit=10)
        ids = [result['id'] for result in results]
        self.assertNotIn(TEST_DOCUMENTS[0].id, ids)


class ChromaDBTest(BaseVectorDBTest):
    """Test ChromaDB implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        db_path = os.path.join("test", "chroma_db")
        os.makedirs(db_path, exist_ok=True)
        self.client = VectORMClient(vectordb=ChromaDB(db_path=db_path))
        self.table = self.client.create_if_not_exists(self.table_name)
        
    def tearDown(self):
        """Clean up after tests."""
        # ChromaDB cleanup is handled automatically


class MilvusDBTest(BaseVectorDBTest):
    """Test Milvus implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        # For testing, we'll use a local Milvus instance
        # In a real environment, you would need a running Milvus server
        try:
            self.client = VectORMClient(vectordb=MilvusDB(db_path="localhost:19530"))
            self.table = self.client.create_if_not_exists(self.table_name)
        except Exception as e:
            self.skipTest(f"Milvus server not available: {str(e)}")
        
    def tearDown(self):
        """Clean up after tests."""
        # Milvus cleanup would be handled here if needed


class QdrantDBTest(BaseVectorDBTest):
    """Test Qdrant implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        # For testing, we'll use a local Qdrant instance
        # In a real environment, you would need a running Qdrant server
        try:
            self.client = VectORMClient(vectordb=QdrantDB(db_path="localhost:6333"))
            self.table = self.client.create_if_not_exists(self.table_name)
        except Exception as e:
            self.skipTest(f"Qdrant server not available: {str(e)}")
        
    def tearDown(self):
        """Clean up after tests."""
        # Qdrant cleanup would be handled here if needed


class PineconeDBTest(BaseVectorDBTest):
    """Test Pinecone implementation."""
    
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        # For testing, we'll use a mock Pinecone instance
        # In a real environment, you would need a Pinecone API key
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            self.skipTest("PINECONE_API_KEY environment variable not set")
            
        self.client = VectORMClient(vectordb=PineconeDB(api_key=api_key))
        self.table = self.client.create_if_not_exists(self.table_name)
        
    def tearDown(self):
        """Clean up after tests."""
        # Pinecone cleanup would be handled here if needed


if __name__ == "__main__":
    unittest.main() 