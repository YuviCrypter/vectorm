from vectordb import VectorDB
from vectordb_table import Table

class VectORMClient:

    vectordb: VectorDB

    def __init__(self, vectordb: VectorDB):
        self.vectordb = vectordb

    def create_if_not_exists(self,table_name: str):
        
        return Table(
            search=self.vectordb.search,
            insert=self.vectordb.insert,
            update=self.vectordb.update,
            delete=self.vectordb.delete,    
        )