from typing import List, Dict, Any, Optional, Union
import uuid
import pinecone

from vectorm.vectordb.base import Document, VectorDB

class PineconeDB(VectorDB):
    """
    Pinecone implementation of the VectorDB interface.
    
    This class provides a concrete implementation of the VectorDB abstract base class
    using Pinecone as the underlying vector database.
    """

    _client = None
    _dimension = 1536  # Default dimension for embeddings

    def __init__(self, api_key: str = None, environment: str = "us-west1-gcp", dimension: int = 1536):
        """
        Initialize a Pinecone instance.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            dimension: Dimension of the vectors to be stored
        """
        self.api_key = api_key
        self.environment = environment
        self._dimension = dimension

        # Initialize Pinecone connection
        if self._client is None and api_key:
            self.connect(api_key)
        
    def connect(self, connection_string: str) -> None:
        """
        Establish a connection to the Pinecone database.
        
        Args:
            connection_string: Pinecone API key
        """
        print("Initializing Pinecone client...")
        pinecone.init(api_key=connection_string, environment=self.environment)
        self._client = pinecone

    def get_table(self, table_name: str) -> Optional[str]:
        """
        Get an index from Pinecone by name.
        
        Args:
            table_name: Name of the index to retrieve
            
        Returns:
            The index name if it exists, None otherwise
        """
        try:
            if table_name in pinecone.list_indexes():
                return table_name
            return None
        except Exception as e:
            print(f"Error getting index: {e}")
            return None

    def create_table(self, table_name: str) -> str:
        """
        Create a new index in Pinecone.
        
        Args:
            table_name: Name of the index to create
            
        Returns:
            The name of the newly created index
        """
        # Create the index
        pinecone.create_index(
            name=table_name,
            dimension=self._dimension,
            metric="cosine"
        )
        
        return table_name

    def insert(self, collection: Union[str, Any], data: List[Document] = None) -> None:
        """
        Insert documents into an index.
        
        Args:
            collection: Name of the index to insert into
            data: List of Document objects to insert
        """
        if data is None:
            data = []
            
        # If collection is a string, get or create the index
        if isinstance(collection, str):
            index_name = self.get_table(collection)
            if index_name is None:
                index_name = self.create_table(collection)
        else:
            # If collection is already an index name, use it directly
            index_name = collection
        
        # Prepare data for insertion
        if data:
            # Get the index
            index = pinecone.Index(index_name)
            
            # Prepare vectors for upsert
            vectors = []
            for doc in data:
                vectors.append({
                    "id": doc.id,
                    "values": doc.vector,
                    "metadata": {
                        "document": doc.document,
                        **doc.metadata
                    }
                })
            
            # Insert the data
            index.upsert(vectors=vectors)

    def search(self, collection: Union[str, Any], query_vector: List[float], 
               limit: int = 10, filter_: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in an index.
        
        Args:
            collection: Name of the index to search in
            query_vector: The vector to search for
            limit: Maximum number of results to return
            filter_: Optional filter criteria to apply to the search
            
        Returns:
            A list of dictionaries containing the search results
        """
        # If collection is a string, get the index
        if isinstance(collection, str):
            index_name = self.get_table(collection)
            if index_name is None:
                return []
        else:
            # If collection is already an index name, use it directly
            index_name = collection
            
        # Get the index
        index = pinecone.Index(index_name)
        
        # Execute the search
        results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True,
            filter=filter_
        )
        
        # Format results to match expected output
        formatted_results = []
        for match in results.matches:
            metadata = match.metadata
            document = metadata.pop("document", "")
            
            formatted_results.append({
                'id': match.id,
                'document': document,
                'metadata': metadata,
                'distance': match.score
            })
                
        return formatted_results

    def delete(self, collection: Union[str, Any], doc_id: str) -> bool:
        """
        Delete a document from an index by its ID.
        
        Args:
            collection: Name of the index to delete from
            doc_id: ID of the document to delete
            
        Returns:
            True if the document was successfully deleted, False otherwise
        """
        try:
            # If collection is a string, get the index
            if isinstance(collection, str):
                index_name = self.get_table(collection)
                if index_name is None:
                    return False
            else:
                # If collection is already an index name, use it directly
                index_name = collection
                
            # Get the index
            index = pinecone.Index(index_name)
            
            # Delete the document
            index.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False 