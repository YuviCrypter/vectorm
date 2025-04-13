from vectorm.vectordb import VectorDB
from vectorm.vectordb_table import Table

class VectORMClient:

    vectordb: VectorDB

    def __init__(self, vectordb: VectorDB):
        self.vectordb = vectordb

    def create_if_not_exists(self,table_name: str):
        
        return Table(
            table_name,
            self.vectordb 
        )