from typing import List, Dict, Any, Optional, Union
from chromadb import Collection

from vectorm.vectordb.base import VectorDB, Document

class Table:
    """
    A wrapper class for vector database tables that provides a simplified interface
    for common operations like search, insert, and delete.
    
    This class abstracts away the collection name parameter by storing it internally,
    allowing for a more intuitive API when working with a specific table.
    """
    
    def __init__(self, table_name: str, vector_db: VectorDB):
        """
        Initialize a Table instance.
        
        Args:
            table_name: The name of the table to work with
            vector_db: An instance of a VectorDB implementation
        """
        self._table_name = table_name
        self._vector_db = vector_db
        self._table: Optional[Collection] = vector_db.get_table(table_name)
        
        # Create the table if it doesn't exist
        if self._table is None:
            self._table = vector_db.create_table(table_name)
            self.table_exists = True
        else:
            self.table_exists = True

    def search(self, query_vector: List[float], limit: int = 10, 
               filter_: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the table.
        
        Args:
            query_vector: The vector to search for
            limit: Maximum number of results to return
            filter_: Optional filter criteria to apply to the search
            
        Returns:
            A list of dictionaries containing the search results
        """
        # Pass the Collection object directly if available, otherwise use the table name
        collection = self._table if self._table is not None else self._table_name
        return self._vector_db.search(
            collection=collection,
            query_vector=query_vector,
            limit=limit,
            filter_=filter_
        )

    def insert(self, data: List[Document]) -> None:
        """
        Insert documents into the table.
        
        Args:
            data: A list of Document objects to insert
        """
        # Pass the Collection object directly if available, otherwise use the table name
        collection = self._table if self._table is not None else self._table_name
        self._vector_db.insert(
            collection=collection,
            data=data
        )

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the table by its ID.
        
        Args:
            doc_id: The ID of the document to delete
            
        Returns:
            True if the document was successfully deleted, False otherwise
        """
        # Pass the Collection object directly if available, otherwise use the table name
        collection = self._table if self._table is not None else self._table_name
        return self._vector_db.delete(
            collection=collection,
            doc_id=doc_id
        )
