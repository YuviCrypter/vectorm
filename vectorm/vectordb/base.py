from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class VectorDB(ABC):
    """Abstract base class for Vector Database"""

    @abstractmethod
    def connect(self, connection_string: str) -> None:
        """Establish a connection to the vector database."""
        raise NotImplementedError

    @abstractmethod
    def insert(self, collection: str, vector: List[float], metadata: Dict[str, Any] = {}) -> Any:
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

    @abstractmethod
    def doc_exists(self, collection: str, doc_id: str) -> bool:
        """Check if a document exists in the collection."""
        raise NotImplementedError

    @abstractmethod
    def create_collection(self, collection: str) -> None:
        """Create a new collection in the vector database."""
        raise NotImplementedError

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections available in the vector database."""
        raise NotImplementedError

    @abstractmethod
    def drop_collection(self, collection: str) -> None:
        """Remove a collection and all its vectors from the database."""
        raise NotImplementedError
