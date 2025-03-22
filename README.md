## Code Use Example

```python
from vectorm import VectORMClient
from vectorm.vectordb import ChromaDB

client = VectORMClient(vectordb=ChromaDB(persist_directory="db"))

table = client.create_if_not_exists("table_name")

table.search()
table.insert()
table.update()
table.delete()
```

## Classes Pseudocode

```python
class ChromaDB(VectorDB):
    persist_directory:str
```

Base Class for vector databases like **ChromaDB**

```python
class VectorDB():

    connection_url:str
    embedding_model:str
    embedding_dimension:int

    def search()
    def insert()
    def update()
    def delete()
```

```python
class VectORMClient(vectordb:VectorDB):

    def create_if_not_exists(table_name:str):
        return Table(
            search=vectordb.search,
            insert=vectordb.insert,
            update=vectordb.update,
            delete=vectordb.delete
        )
```

```python
class Table(search, insert, update, delete):

    def search()
    def insert()
    def update()
    def delete()

    self.search = search
    self.insert = insert
    self.update = update
    self.delete = delete
```
