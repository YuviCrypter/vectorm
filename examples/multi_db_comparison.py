import os
import time
import uuid
import numpy as np
from typing import List, Dict, Any

from vectorm import VectORMClient
from vectorm.vectordb import ChromaDB, MilvusDB, QdrantDB, PineconeDB, Document

def create_documents(count: int = 100, dimension: int = 1536) -> List[Document]:
    """
    Create a list of test documents.
    
    Args:
        count: Number of documents to create
        dimension: Dimension of the vectors
        
    Returns:
        A list of Document objects
    """
    documents = []
    for i in range(count):
        documents.append(
            Document(
                id=str(uuid.uuid4()),
                document=f"This is test document {i} for vector database comparison.",
                vector=np.random.rand(dimension).tolist(),
                metadata={"index": i, "source": "test", "category": "comparison"}
            )
        )
    return documents

def benchmark_insert(table, documents: List[Document]) -> float:
    """
    Benchmark document insertion.
    
    Args:
        table: The table to insert into
        documents: List of documents to insert
        
    Returns:
        Time taken for insertion in seconds
    """
    start_time = time.time()
    table.insert(documents)
    end_time = time.time()
    return end_time - start_time

def benchmark_search(table, query_vector: List[float], limit: int = 10) -> float:
    """
    Benchmark vector search.
    
    Args:
        table: The table to search in
        query_vector: The query vector
        limit: Maximum number of results to return
        
    Returns:
        Time taken for search in seconds
    """
    start_time = time.time()
    results = table.search(query_vector, limit=limit)
    end_time = time.time()
    return end_time - start_time, results

def main():
    """
    Compare performance of different vector databases.
    """
    # Create test data
    print("Creating test data...")
    documents = create_documents(count=100)
    query_vector = np.random.rand(1536).tolist()
    
    # Test ChromaDB
    print("\n=== Testing ChromaDB ===")
    db_path = os.path.join("data", "chroma_db")
    os.makedirs(db_path, exist_ok=True)
    chroma_client = VectORMClient(vectordb=ChromaDB(db_path=db_path))
    chroma_table = chroma_client.create_if_not_exists("benchmark_chroma")
    
    insert_time = benchmark_insert(chroma_table, documents)
    print(f"ChromaDB insert time: {insert_time:.4f} seconds")
    
    search_time, results = benchmark_search(chroma_table, query_vector)
    print(f"ChromaDB search time: {search_time:.4f} seconds")
    print(f"ChromaDB found {len(results)} results")
    
    # Test Milvus (if available)
    try:
        print("\n=== Testing Milvus ===")
        milvus_client = VectORMClient(vectordb=MilvusDB(db_path="localhost:19530"))
        milvus_table = milvus_client.create_if_not_exists("benchmark_milvus")
        
        insert_time = benchmark_insert(milvus_table, documents)
        print(f"Milvus insert time: {insert_time:.4f} seconds")
        
        search_time, results = benchmark_search(milvus_table, query_vector)
        print(f"Milvus search time: {search_time:.4f} seconds")
        print(f"Milvus found {len(results)} results")
    except Exception as e:
        print(f"Milvus test skipped: {e}")
    
    # Test Qdrant (if available)
    try:
        print("\n=== Testing Qdrant ===")
        qdrant_client = VectORMClient(vectordb=QdrantDB(db_path="localhost:6333"))
        qdrant_table = qdrant_client.create_if_not_exists("benchmark_qdrant")
        
        insert_time = benchmark_insert(qdrant_table, documents)
        print(f"Qdrant insert time: {insert_time:.4f} seconds")
        
        search_time, results = benchmark_search(qdrant_table, query_vector)
        print(f"Qdrant search time: {search_time:.4f} seconds")
        print(f"Qdrant found {len(results)} results")
    except Exception as e:
        print(f"Qdrant test skipped: {e}")
    
    # Test Pinecone (if API key is available)
    api_key = os.environ.get("PINECONE_API_KEY")
    if api_key:
        try:
            print("\n=== Testing Pinecone ===")
            pinecone_client = VectORMClient(vectordb=PineconeDB(api_key=api_key))
            pinecone_table = pinecone_client.create_if_not_exists("benchmark_pinecone")
            
            insert_time = benchmark_insert(pinecone_table, documents)
            print(f"Pinecone insert time: {insert_time:.4f} seconds")
            
            search_time, results = benchmark_search(pinecone_table, query_vector)
            print(f"Pinecone search time: {search_time:.4f} seconds")
            print(f"Pinecone found {len(results)} results")
        except Exception as e:
            print(f"Pinecone test skipped: {e}")
    else:
        print("\nPinecone test skipped: PINECONE_API_KEY environment variable not set")

if __name__ == "__main__":
    main() 