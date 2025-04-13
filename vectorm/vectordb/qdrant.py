from typing import List, Dict, Any, Optional, Union
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from vectorm.vectordb.base import Document, VectorDB

class QdrantDB(VectorDB):
    """
    Qdrant implementation of the VectorDB interface.
    
    This class provides a concrete implementation of the VectorDB abstract base class
    using Qdrant as the underlying vector database.
    """

    _client = None
    _dimension = 1536  # Default dimension for embeddings

    def __init__(self, db_path: str = "localhost:6333", dimension: int = 1536):
        """
        Initialize a Qdrant instance.
        
        Args:
            db_path: Connection string for Qdrant (host:port)
            dimension: Dimension of the vectors to be stored
        """
        self.db_path = db_path
        self._dimension = dimension

        # Initialize Qdrant connection
        if self._client is None:
            self.connect(self.db_path)
        
    def connect(self, connection_string: str) -> None:
        """
        Establish a connection to the Qdrant database.
        
        Args:
            connection_string: Connection string for Qdrant (host:port)
        """
        print("Initializing Qdrant client...")
        host, port = connection_string.split(":")
        self._client = QdrantClient(host=host, port=int(port))

    def get_table(self, table_name: str) -> Optional[Any]:
        """
        Get a collection from Qdrant by name.
        
        Args:
            table_name: Name of the collection to retrieve
            
        Returns:
            The collection name if it exists, None otherwise
        """
        try:
            collections = self._client.get_collections().collections
            for collection in collections:
                if collection.name == table_name:
                    return table_name
            return None
        except Exception as e:
            print(f"Error getting collection: {e}")
            return None

    def create_table(self, table_name: str) -> str:
        """
        Create a new collection in Qdrant.
        
        Args:
            table_name: Name of the collection to create
            
        Returns:
            The name of the newly created collection
        """
        # Create the collection
        self._client.create_collection(
            collection_name=table_name,
            vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE)
        )
        
        return table_name

    def insert(self, collection: Union[str, Any], data: List[Document] = None) -> None:
        """
        Insert documents into a collection.
        
        Args:
            collection: Name of the collection to insert into
            data: List of Document objects to insert
        """
        if data is None:
            data = []
            
        # If collection is a string, get or create the table
        if isinstance(collection, str):
            table_name = self.get_table(collection)
            if table_name is None:
                table_name = self.create_table(collection)
        else:
            # If collection is already a collection name, use it directly
            table_name = collection
        
        # Prepare data for insertion
        if data:
            points = []
            for doc in data:
                points.append(models.PointStruct(
                    id=doc.id,
                    vector=doc.vector,
                    payload={
                        "document": doc.document,
                        "metadata": doc.metadata
                    }
                ))
            
            # Insert the data
            self._client.upsert(
                collection_name=table_name,
                points=points
            )

    def search(self, collection: Union[str, Any], query_vector: List[float], 
               limit: int = 10, filter_: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection: Name of the collection to search in
            query_vector: The vector to search for
            limit: Maximum number of results to return
            filter_: Optional filter criteria to apply to the search
            
        Returns:
            A list of dictionaries containing the search results
        """
        # If collection is a string, get the table
        if isinstance(collection, str):
            table_name = self.get_table(collection)
            if table_name is None:
                return []
        else:
            # If collection is already a collection name, use it directly
            table_name = collection
            
        # Execute the search
        search_result = self._client.search(
            collection_name=table_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=models.Filter(**filter_) if filter_ else None
        )
        
        # Format results to match expected output
        formatted_results = []
        for scored_point in search_result:
            formatted_results.append({
                'id': scored_point.id,
                'document': scored_point.payload.get('document'),
                'metadata': scored_point.payload.get('metadata'),
                'distance': scored_point.score
            })
                
        return formatted_results

    def delete(self, collection: Union[str, Any], doc_id: str) -> bool:
        """
        Delete a document from a collection by its ID.
        
        Args:
            collection: Name of the collection to delete from
            doc_id: ID of the document to delete
            
        Returns:
            True if the document was successfully deleted, False otherwise
        """
        try:
            # If collection is a string, get the table
            if isinstance(collection, str):
                table_name = self.get_table(collection)
                if table_name is None:
                    return False
            else:
                # If collection is already a collection name, use it directly
                table_name = collection
                
            # Delete the document
            self._client.delete(
                collection_name=table_name,
                points_selector=models.PointIdsList(points=[doc_id])
            )
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False 