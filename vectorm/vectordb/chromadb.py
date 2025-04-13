from typing import List, Dict, Any, Optional, Union
from vectorm.vectordb.base import Document, VectorDB

from chromadb import PersistentClient, Collection
from chromadb.api.client import ClientAPI

class ChromaDB(VectorDB):
    """
    ChromaDB implementation of the VectorDB interface.
    
    This class provides a concrete implementation of the VectorDB abstract base class
    using ChromaDB as the underlying vector database.
    """

    _client: Optional[ClientAPI] = None

    def __init__(self, db_path: str = "chroma_db"):
        """
        Initialize a ChromaDB instance.
        
        Args:
            db_path: Path to the ChromaDB database directory
        """
        self.db_path = db_path

        # Initialize ChromaDB connection
        if self._client is None:
            self.connect(self.db_path)
        
    def connect(self, db_path: str) -> None:
        """
        Establish a connection to the ChromaDB database.
        
        Args:
            db_path: Path to the ChromaDB database directory
        """
        print("Initializing ChromaDB client...")
        self._client = PersistentClient(path=db_path)

    def get_table(self, table_name: str) -> Optional[Collection]:
        """
        Get a collection from ChromaDB by name.
        
        Args:
            table_name: Name of the collection to retrieve
            
        Returns:
            The collection if it exists, None otherwise
        """
        try:
            return self._client.get_collection(table_name)
        except Exception as e:
            return None

    def create_table(self, table_name: str) -> Collection:
        """
        Create a new collection in ChromaDB.
        
        Args:
            table_name: Name of the collection to create
            
        Returns:
            The newly created collection
        """
        return self._client.create_collection(table_name)

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
        
        # Batch insert for better performance
        if data:
            ids = [doc.id for doc in data]
            embeddings = [doc.vector for doc in data]
            documents = [doc.document for doc in data]
            metadatas = [doc.metadata for doc in data]
            
            table.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

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
            
        results = table.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=filter_
        )
        
        # Format results to match expected output
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
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
                
            table.delete(ids=[doc_id])
            return True
        except Exception:
            return False