from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Remove the circular import
# from vectorm.vectordb.base import Document

class VectorDB(ABC):
    """Abstract base class for Vector Database"""

    _client: Any

    @abstractmethod
    def connect(self, connection_string: str) -> None:
        """Establish a connection to the vector database."""
        raise NotImplementedError

    @abstractmethod
    def insert(self, collection: str, data: List['Document'] = {}) -> Any:
        """Insert a vector with metadata into the specified collection."""
        raise NotImplementedError

    @abstractmethod
    def search(self, collection: str, query_vector: List[float], limit: int = 10, filter_: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in the collection."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, collection: str, doc_id: str) -> bool:
        """Delete a vector entry from the collection by its document ID."""
        raise NotImplementedError

    # @abstractmethod
    # def doc_exists(self, collection: str, doc_id: str) -> bool:
    #     """Check if a document exists in the collection."""
    #     raise NotImplementedError

    @abstractmethod
    def create_table(self, table_name: str) -> None:
        """Create a new table in the vector database."""
        raise NotImplementedError
    
    @abstractmethod
    def get_table(self, table_name: str):
        """Get a table from the vector database."""
        raise NotImplementedError

    # @abstractmethod
    # def list_collections(self) -> List[str]:
    #     """List all collections available in the vector database."""
    #     raise NotImplementedError

    # @abstractmethod
    # def drop_collection(self, collection: str) -> None:
    #     """Remove a collection and all its vectors from the database."""
    #     raise NotImplementedError


class Document:
    """A class representing a document with a vector and metadata."""

    id: str
    document: str
    vector: List[float]  # Changed from float to List[float] to match usage
    metadata: Dict[str, Any]
    
    def __init__(self, id: str, document: str, vector: List[float], metadata: Dict[str, Any] = None):
        """Initialize a Document with id, document text, vector, and optional metadata."""
        self.id = id
        self.document = document
        self.vector = vector
        self.metadata = metadata or {}