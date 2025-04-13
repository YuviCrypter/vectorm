from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vectorm",
    version="0.1.0",
    author="Yuvi",
    author_email="yuvrajdarshankar@gmail.com",
    description="A unified interface for vector databases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuvrajdarshankar/vectorm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "chromadb": ["chromadb>=0.3.0"],
        "milvus": ["pymilvus>=2.0.0"],
        "qdrant": ["qdrant-client>=1.0.0"],
        "pinecone": ["pinecone-client>=2.0.0"],
        "lancedb": ["lancedb>=0.4.0"],
        "all": [
            "chromadb>=0.3.0",
            "pymilvus>=2.0.0",
            "qdrant-client>=1.0.0",
            "pinecone-client>=2.0.0",
            "lancedb>=0.4.0",
        ],
    },
) 