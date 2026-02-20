"""
RAG System for Buddha AI
Vector database using ChromaDB for semantic retrieval of Buddhist texts.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
from pathlib import Path

from text_chunker import TextChunker


class BuddhaRAG:
    """RAG system for retrieving relevant Buddhist text passages."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_dir: str = "data/chroma"):
        """
        Initialize the RAG system.

        Args:
            model_name: Sentence transformer model to use for embeddings
            persist_dir: Directory to persist the vector database
        """
        self.model_name = model_name
        self.persist_dir = persist_dir

        # Initialize sentence transformer model
        print(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        # Initialize ChromaDB client
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Collection name
        self.collection_name = "buddha_texts"
        self.collection = None

    def initialize_database(self, force_rebuild: bool = False):
        """
        Initialize or load the vector database.

        Args:
            force_rebuild: If True, delete existing collection and rebuild
        """
        # Check if collection exists
        try:
            if force_rebuild:
                print("Deleting existing collection...")
                self.client.delete_collection(self.collection_name)
                raise ValueError("Force rebuild")

            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            count = self.collection.count()
            print(f"Loaded existing collection with {count} chunks")

        except:
            # Create new collection
            print("Creating new collection...")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Buddhist texts for Buddha AI"}
            )

            # Load and process texts
            print("Processing texts...")
            chunker = TextChunker(chunk_size=300, overlap=50)
            chunks = chunker.process_all_texts()

            # Add chunks to database
            self._add_chunks_to_db(chunks)

    def _add_chunks_to_db(self, chunks: List[Dict]):
        """
        Add text chunks to the vector database.

        Args:
            chunks: List of chunk dictionaries from TextChunker
        """
        print(f"\nAdding {len(chunks)} chunks to vector database...")

        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i}")
            texts.append(chunk['text'])
            metadatas.append(chunk['metadata'])

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end]
            )
            print(f"  Added batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

        print(f"Database initialized with {len(chunks)} chunks")

    def query(self, query_text: str, n_results: int = 3, topic_filter: str = None) -> List[Dict]:
        """
        Query the vector database for relevant passages.

        Args:
            query_text: The query string
            n_results: Number of results to return
            topic_filter: Optional topic to filter by

        Returns:
            List of relevant text chunks with metadata
        """
        if not self.collection:
            raise ValueError("Database not initialized. Call initialize_database() first.")

        # Build where clause for topic filtering
        where = None
        if topic_filter:
            where = {"topics": {"$contains": topic_filter}}

        # Query the database
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )

        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })

        return formatted_results

    def get_relevant_passages(self, topic: str, user_input: str = "", n_results: int = 3) -> str:
        """
        Get relevant passages for a topic and optional user input.

        Args:
            topic: The philosophical topic being discussed
            user_input: Optional user input to make retrieval more relevant
            n_results: Number of passages to retrieve

        Returns:
            Formatted string of relevant passages for prompt
        """
        # Map topics to keywords
        topic_keywords = {
            'four_noble_truths': ['suffering', 'dukkha', 'noble truths', 'craving', 'tanha'],
            'middle_way': ['middle way', 'middle path', 'extremes', 'moderation'],
            'impermanence': ['impermanence', 'anicca', 'change', 'flux', 'transient'],
            'non_self': ['anatta', 'non-self', 'no-self', 'self', 'soul', 'atman']
        }

        # Get keywords for this topic
        keywords = topic_keywords.get(topic, [])

        # Construct query
        if user_input:
            # Combine topic keywords with user input for more relevant retrieval
            query = f"{' '.join(keywords[:2])} {user_input}"
        else:
            query = ' '.join(keywords)

        # Query database
        results = self.query(query, n_results=n_results)

        # Format for prompt
        if not results:
            return "No relevant passages found."

        formatted = "RELEVANT PASSAGES FROM BUDDHA'S TEACHINGS:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"Passage {i} (from {result['metadata']['source']}):\n"
            formatted += f"{result['text']}\n\n"

        return formatted


def test_rag_system():
    """Test the RAG system."""
    print("=" * 60)
    print("Testing Buddha RAG System")
    print("=" * 60)

    # Initialize RAG
    rag = BuddhaRAG()
    rag.initialize_database(force_rebuild=False)

    # Test queries
    test_queries = [
        ("What is suffering?", None),
        ("Tell me about impermanence", None),
        ("What is the self?", None),
        ("How do I find peace?", None),
    ]

    print("\n" + "=" * 60)
    print("Test Queries:")
    print("=" * 60)

    for query, topic in test_queries:
        print(f"\n\nQuery: '{query}'")
        print("-" * 60)

        results = rag.query(query, n_results=2)

        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Source: {result['metadata']['source']}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Text: {result['text'][:300]}...")

    # Test topic-based retrieval
    print("\n\n" + "=" * 60)
    print("Test Topic-Based Retrieval:")
    print("=" * 60)

    for topic in ['four_noble_truths', 'impermanence', 'non_self']:
        print(f"\nTopic: {topic}")
        print("-" * 60)
        passages = rag.get_relevant_passages(topic, n_results=2)
        print(passages[:500] + "...\n")

    return rag


if __name__ == "__main__":
    test_rag_system()
