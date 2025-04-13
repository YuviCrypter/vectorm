from typing import List, Dict, Any, Optional, Union
import lancedb
from lancedb import LanceDBConnection
from lancedb.table import LanceTable
import pyarrow as pa
from vectorm.vectordb.base import Document, VectorDB

class LanceDB(VectorDB):
    """
    LanceDB implementation of the VectorDB interface.
    
    This class provides a concrete implementation of the VectorDB abstract base class
    using LanceDB as the underlying vector database.
    """

    _client: Optional[LanceDBConnection] = None
    _vector_dimension: int = 1536  # Default dimension, can be overridden
    _base_schema: pa.Schema = None  # Store the base schema

    def __init__(self, db_path: str = "lancedb", vector_dimension: int = 1536, sample_metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a LanceDB instance.
        
        Args:
            db_path: Path to the LanceDB database directory
            vector_dimension: Dimension of the vector embeddings
            sample_metadata: Optional sample metadata to infer schema from
        """
        self.db_path = db_path
        self._vector_dimension = vector_dimension

        # Create the base schema
        self._base_schema = self._create_pyarrow_schema(sample_metadata)

        # Initialize LanceDB connection
        if self._client is None:
            self.connect(self.db_path)
        
    def connect(self, db_path: str) -> None:
        """
        Establish a connection to the LanceDB database.
        
        Args:
            db_path: Path to the LanceDB database directory
        """
        print("Initializing LanceDB client...")
        self._client = lancedb.connect(db_path)

    def get_table(self, table_name: str) -> Optional[LanceTable]:
        """
        Get a table from LanceDB by name.
        
        Args:
            table_name: Name of the table to retrieve
            
        Returns:
            The table if it exists, None otherwise
        """
        try:
            return self._client.open_table(table_name)
        except Exception as e:
            return None

    def _infer_pyarrow_type(self, value: Any) -> pa.DataType:
        """
        Infer the PyArrow data type from a Python value.
        
        Args:
            value: The value to infer the type from
            
        Returns:
            The PyArrow data type
        """
        if isinstance(value, bool):
            return pa.bool_()
        elif isinstance(value, int):
            return pa.int64()
        elif isinstance(value, float):
            return pa.float64()
        elif isinstance(value, str):
            return pa.string()
        elif isinstance(value, list):
            if value and isinstance(value[0], float):
                return pa.list_(pa.float32(), self._vector_dimension)
            elif value and isinstance(value[0], int):
                return pa.list_(pa.int32(), self._vector_dimension)
            else:
                return pa.string()  # Default to string for other list types
        else:
            return pa.string()  # Default to string for other types

    def _create_pyarrow_schema(self, metadata: Dict[str, Any] = None) -> pa.Schema:
        """
        Create a PyArrow schema for a LanceDB table.
        
        Args:
            metadata: Dictionary of metadata key-value pairs
            
        Returns:
            PyArrow schema
        """
        # Start with the base schema
        fields = [
            pa.field("id", pa.string()),
            pa.field("document", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self._vector_dimension))
        ]
        
        # Add metadata fields if provided
        if metadata:
            for key, value in metadata.items():
                fields.append(pa.field(key, self._infer_pyarrow_type(value)))
        
        return pa.schema(fields)

    def create_table(self, table_name: str) -> LanceTable:
        """
        Create a new table in LanceDB.
        
        Args:
            table_name: Name of the table to create
            
        Returns:
            The newly created table
        """
        # Use the pre-defined schema
        return self._client.create_table(table_name, schema=self._base_schema)

    def insert(self, collection: Union[str, LanceTable], data: List[Document] = None) -> None:
        """
        Insert documents into a table.
        
        Args:
            collection: Name of the table or a LanceTable object to insert into
            data: List of Document objects to insert
        """
        if data is None:
            data = []
            
        # If collection is a string, get or create the table
        if isinstance(collection, str):
            table = self.get_table(collection)
            if table is None:
                # Create table with the pre-defined schema
                table = self.create_table(collection)
        else:
            # If collection is already a LanceTable object, use it directly
            table = collection
        
        # Prepare data for insertion
        if data:
            # Convert documents to the format expected by LanceDB
            records = []
            for doc in data:
                # Start with the base fields
                record = {
                    "id": doc.id,
                    "document": doc.document,
                    "vector": doc.vector,
                }
                
                # Add metadata fields as separate columns
                if doc.metadata:
                    for key, value in doc.metadata.items():
                        record[key] = value
                
                records.append(record)
            
            # Insert the records
            table.add(records)

    def search(self, collection: Union[str, LanceTable], query_vector: List[float], 
               limit: int = 10, filter_: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a table.
        
        Args:
            collection: Name of the table or a LanceTable object to search in
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
            # If collection is already a LanceTable object, use it directly
            table = collection
            
        # Build the search query
        search_query = table.search(query_vector).limit(limit)
        
        # Apply filters if provided
        if filter_:
            filter_conditions = []
            for key, value in filter_.items():
                if isinstance(value, (int, float, bool, str)):
                    filter_conditions.append(f"{key} = {repr(value)}")
                elif isinstance(value, list):
                    filter_conditions.append(f"{key} IN {repr(value)}")
                else:
                    filter_conditions.append(f"{key} = '{str(value)}'")
            
            if filter_conditions:
                search_query = search_query.filter(" AND ".join(filter_conditions))
        
        # Execute the search
        results = search_query.to_list()
        
        # Format results to match expected output
        formatted_results = []
        for result in results:
            # Extract base fields
            formatted_result = {
                'id': result['id'],
                'document': result['document'],
                'metadata': {},
                'distance': result['_distance'] if '_distance' in result else None
            }
            
            # Extract metadata fields (all fields except id, document, vector, and _distance)
            for key, value in result.items():
                if key not in ['id', 'document', 'vector', '_distance']:
                    formatted_result['metadata'][key] = value
            
            formatted_results.append(formatted_result)
            
        return formatted_results

    def delete(self, collection: Union[str, LanceTable], doc_id: str) -> bool:
        """
        Delete a document from a table by its ID.
        
        Args:
            collection: Name of the table or a LanceTable object to delete from
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
                # If collection is already a LanceTable object, use it directly
                table = collection
                
            # Delete the record
            table.delete(f"id = '{doc_id}'")
            return True
        except Exception:
            return False 