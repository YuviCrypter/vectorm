import os
import uuid
import numpy as np
from vectorm import VectORMClient
from vectorm.vectordb import Document, LanceDB

def main():
    """
    Basic example of using VectORM with LanceDB.
    """
    # Create a client with LanceDB
    db_path = os.path.join("data", "lancedb")
    os.makedirs(db_path, exist_ok=True)
    client = VectORMClient(vectordb=LanceDB(db_path=db_path, vector_dimension=1536, sample_metadata={"source": "example", "category": "database"}))
    
    # Create a table
    table = client.create_if_not_exists("example_table")
    print(f"Table created: {table.table_exists}")
    
    # Create some example documents
    documents = [
        Document(
            id=str(uuid.uuid4()),
            document="Vector databases are used for similarity search.",
            vector=np.random.rand(1536).tolist(),
            metadata={"source": "example", "category": "database"}
        ),
        Document(
            id=str(uuid.uuid4()),
            document="ChromaDB is a popular vector database.",
            vector=np.random.rand(1536).tolist(),
            metadata={"source": "example", "category": "chromadb"}
        ),
        Document(
            id=str(uuid.uuid4()),
            document="Vector similarity search is useful for recommendation systems.",
            vector=np.random.rand(1536).tolist(),
            metadata={"source": "example", "category": "recommendation"}
        )
    ]
    
    # Insert documents
    table.insert(documents)
    print(f"Inserted {len(documents)} documents")
    
    # Create a query vector
    query_vector = np.random.rand(1536).tolist()
    
    # Search for similar vectors
    results = table.search(query_vector, limit=2)
    print(f"Found {len(results)} similar documents")
    
    # Print results
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"ID: {result['id']}")
        print(f"Document: {result['document']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Distance: {result['distance']}")
    
    # Delete a document
    if results:
        doc_id = results[0]['id']
        success = table.delete(doc_id)
        print(f"\nDeleted document {doc_id}: {success}")

if __name__ == "__main__":
    main() 