# VectORM

VectORM is a Python library that provides a unified interface for working with various vector databases. It abstracts away the differences between different vector database implementations, allowing you to easily switch between them without changing your application code.

## Features

- **Unified API**: Use the same interface for all supported vector databases
- **Multiple Backends**: Support for ChromaDB, Milvus, Qdrant, and Pinecone
- **Simple Interface**: Easy-to-use API for common vector database operations
- **Type Hints**: Full type hints for better IDE support and code quality
- **Comprehensive Documentation**: Detailed docstrings for all classes and methods

## Installation

```bash
pip install vectorm
```

### Dependencies

VectORM requires the following dependencies:

- Python 3.7+
- ChromaDB (for ChromaDB backend)
- Pymilvus (for Milvus backend)
- Qdrant Client (for Qdrant backend)
- Pinecone Client (for Pinecone backend)
- NumPy (for vector operations)

You can install the dependencies for all backends with:

```bash
pip install vectorm[all]
```

Or install specific backends:

```bash
pip install vectorm[chromadb]  # For ChromaDB only
pip install vectorm[milvus]    # For Milvus only
pip install vectorm[qdrant]    # For Qdrant only
pip install vectorm[pinecone]  # For Pinecone only
```

## Usage

### Basic Usage

```python
from vectorm import VectORMClient
from vectorm.vectordb import ChromaDB, Document
import numpy as np
import uuid

# Create a client with ChromaDB
client = VectORMClient(vectordb=ChromaDB(db_path="data/chroma_db"))

# Create a table
table = client.create_if_not_exists("my_table")

# Create a document
document = Document(
    id=str(uuid.uuid4()),
    document="This is a test document.",
    vector=np.random.rand(1536).tolist(),
    metadata={"source": "test"}
)

# Insert a document
table.insert([document])

# Search for similar vectors
query_vector = np.random.rand(1536).tolist()
results = table.search(query_vector, limit=5)

# Print results
for result in results:
    print(f"ID: {result['id']}")
    print(f"Document: {result['document']}")
    print(f"Distance: {result['distance']}")
    print("---")

# Delete a document
table.delete(document.id)
```

### Using Different Vector Databases

#### ChromaDB

```python
from vectorm import VectORMClient
from vectorm.vectordb import ChromaDB

client = VectORMClient(vectordb=ChromaDB(db_path="data/chroma_db"))
```

#### Milvus

```python
from vectorm import VectORMClient
from vectorm.vectordb import MilvusDB

client = VectORMClient(vectordb=MilvusDB(db_path="localhost:19530"))
```

#### Qdrant

```python
from vectorm import VectORMClient
from vectorm.vectordb import QdrantDB

client = VectORMClient(vectordb=QdrantDB(db_path="localhost:6333"))
```

#### Pinecone

```python
from vectorm import VectORMClient
from vectorm.vectordb import PineconeDB

client = VectORMClient(vectordb=PineconeDB(api_key="your-api-key"))
```

## API Reference

### VectORMClient

The main client class for interacting with vector databases.

```python
client = VectORMClient(vectordb=ChromaDB(db_path="data/chroma_db"))
table = client.create_if_not_exists("my_table")
```

### Table

A wrapper class for vector database tables.

```python
# Insert documents
table.insert([document1, document2])

# Search for similar vectors
results = table.search(query_vector, limit=10)

# Delete a document
table.delete(document_id)
```

### Document

A class representing a document with a vector and metadata.

```python
document = Document(
    id="unique-id",
    document="Document text",
    vector=[0.1, 0.2, ...],
    metadata={"key": "value"}
)
```

## Testing

Run the tests with:

```bash
python -m unittest discover tests
```

## Examples

Check out the `examples` directory for more usage examples:

- `basic_usage.py`: Basic usage of VectORM with ChromaDB
- `multi_db_comparison.py`: Comparing performance of different vector databases

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
