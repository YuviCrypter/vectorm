from typing import List, Dict, Any, Optional, Union
import uuid
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from vectorm.vectordb.base import Document, VectorDB

class MilvusDB(VectorDB):
    """
    Milvus implementation of the VectorDB interface.
    
    This class provides a concrete implementation of the VectorDB abstract base class
    using Milvus as the underlying vector database.
    """

    _client = None
    _dimension = 1536  # Default dimension for embeddings

    def __init__(self, connection_string: str = "localhost:19530", dimension: int = 1536):
        """
        Initialize a Milvus instance.
        
        Args:
            connection_string: Connection string for Milvus (host:port)
            dimension: Dimension of the vectors to be stored
        """
        self.connection_string = connection_string
        self._dimension = dimension

        # Initialize Milvus connection
        if self._client is None:
            self.connect(self.connection_string)
        
    def connect(self, connection_string: str) -> None:
        """
        Establish a connection to the Milvus database.
        
        Args:
            connection_string: Connection string for Milvus (host:port)
        """
        print("Initializing Milvus client...")
        host, port = connection_string.split(":")
        connections.connect(host=host, port=port)
        self._client = connections

    def get_table(self, table_name: str) -> Optional[Collection]:
        """
        Get a collection from Milvus by name.
        
        Args:
            table_name: Name of the collection to retrieve
            
        Returns:
            The collection if it exists, None otherwise
        """
        try:
            if utility.has_collection(table_name):
                return Collection(table_name)
            return None
        except Exception as e:
            print(f"Error getting collection: {e}")
            return None

    def create_table(self, table_name: str) -> Collection:
        """
        Create a new collection in Milvus.
        
        Args:
            table_name: Name of the collection to create
            
        Returns:
            The newly created collection
        """
        # Define the schema for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        schema = CollectionSchema(fields=fields, description=f"Collection for {table_name}")
        
        # Create the collection
        collection = Collection(name=table_name, schema=schema)
        
        # Create an index for the vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        
        return collection

    def insert(self, collection: Union[str, Collection], data: List[Document] = None) -> None:
        """
        Insert documents into a collection.
        
        Args:
            collection: Name of the collection or a Collection object to insert into
            data: List of Document objects to insert
        """
        if data is None:
            data = []
            
        # If collection is a string, get or create the table
        if isinstance(collection, str):
            table = self.get_table(collection)
            if table is None:
                table = self.create_table(collection)
        else:
            # If collection is already a Collection object, use it directly
            table = collection
        
        # Prepare data for insertion
        if data:
            ids = [doc.id for doc in data]
            documents = [doc.document for doc in data]
            vectors = [doc.vector for doc in data]
            metadatas = [doc.metadata for doc in data]
            
            # Insert the data
            table.insert([
                ids,
                documents,
                vectors,
                metadatas
            ])
            
            # Flush to make the data available for search
            table.flush()

    def search(self, collection: Union[str, Collection], query_vector: List[float], 
               limit: int = 10, filter_: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection: Name of the collection or a Collection object to search in
            query_vector: The vector to search for
            limit: Maximum number of results to return
            filter_: Optional filter criteria to apply to the search
            
        Returns:
            A list of dictionaries containing the search results
        """
        # If collection is a string, get the table
        if isinstance(collection, str):
            table = self.get_table(collection)
            if table is None:
                return []
        else:
            # If collection is already a Collection object, use it directly
            table = collection
            
        # Load the collection into memory
        table.load()
        
        # Prepare search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Execute the search
        results = table.search(
            data=[query_vector],
            ann_field="vector",
            param=search_params,
            limit=limit,
            expr=filter_ if filter_ else None,
            output_fields=["document", "metadata"]
        )
        
        # Format results to match expected output
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    'id': hit.entity.get('id'),
                    'document': hit.entity.get('document'),
                    'metadata': hit.entity.get('metadata'),
                    'distance': hit.distance
                })
                
        return formatted_results

    def delete(self, collection: Union[str, Collection], doc_id: str) -> bool:
        """
        Delete a document from a collection by its ID.
        
        Args:
            collection: Name of the collection or a Collection object to delete from
            doc_id: ID of the document to delete
            
        Returns:
            True if the document was successfully deleted, False otherwise
        """
        try:
            # If collection is a string, get the table
            if isinstance(collection, str):
                table = self.get_table(collection)
                if table is None:
                    return False
            else:
                # If collection is already a Collection object, use it directly
                table = collection
                
            # Delete the document
            expr = f'id == "{doc_id}"'
            table.delete(expr)
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False 