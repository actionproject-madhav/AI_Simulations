"""
Text Chunker for Buddha AI RAG System
Loads Buddhist texts and splits them into overlapping chunks for vector database.
"""

import os
from pathlib import Path
from typing import List, Dict
import re


class TextChunker:
    """Chunks text files into overlapping segments for RAG retrieval."""

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target number of words per chunk
            overlap: Number of words to overlap between chunks (maintains context)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = []

    def load_texts(self, texts_dir: str = "texts") -> Dict[str, str]:
        """
        Load all .txt files from the texts directory.

        Returns:
            Dictionary mapping filename to text content
        """
        texts = {}
        texts_path = Path(texts_dir)

        for txt_file in texts_path.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    texts[txt_file.stem] = content
                    print(f"Loaded {txt_file.name}: {len(content.split())} words")
            except Exception as e:
                print(f"Error loading {txt_file.name}: {e}")

        return texts

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing excessive whitespace and formatting issues.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove page numbers and common artifacts
        text = re.sub(r'\n\d+\n', '\n', text)

        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove leading/trailing whitespace from lines
        text = '\n'.join(line.strip() for line in text.split('\n'))

        return text.strip()

    def chunk_text(self, text: str, source: str, topic_keywords: List[str] = None) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata.

        Args:
            text: Text to chunk
            source: Source document name
            topic_keywords: Optional keywords for this text

        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Clean the text first
        text = self.clean_text(text)

        # Split into words
        words = text.split()
        chunks = []

        # Create overlapping chunks
        start = 0
        chunk_id = 0

        while start < len(words):
            # Get chunk of words
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            # Create metadata
            metadata = {
                'source': source,
                'chunk_id': chunk_id,
                'word_count': len(chunk_words),
                'start_position': start,
            }

            if topic_keywords:
                metadata['topics'] = topic_keywords

            chunks.append({
                'text': chunk_text,
                'metadata': metadata
            })

            chunk_id += 1

            # Move start forward (with overlap)
            if end >= len(words):
                break
            start += (self.chunk_size - self.overlap)

        return chunks

    def process_all_texts(self) -> List[Dict]:
        """
        Load and chunk all texts in the texts directory.

        Returns:
            List of all chunks with metadata
        """
        texts = self.load_texts()
        all_chunks = []

        # Define topic keywords for different sources
        topic_mapping = {
            'Dhammapada': ['four noble truths', 'suffering', 'middle way',
                          'impermanence', 'non-self', 'dharma', 'mindfulness'],
            'sayings': ['four noble truths', 'suffering', 'middle way',
                       'impermanence', 'non-self', 'teaching', 'wisdom']
        }

        for source_name, text in texts.items():
            print(f"\nChunking {source_name}...")

            # Get topic keywords for this source
            keywords = topic_mapping.get(source_name, [])

            # Chunk the text
            chunks = self.chunk_text(text, source_name, keywords)
            all_chunks.extend(chunks)

            print(f"  Created {len(chunks)} chunks")

        self.chunks = all_chunks
        print(f"\nTotal chunks: {len(all_chunks)}")

        return all_chunks

    def get_chunks_by_topic(self, topic_keywords: List[str]) -> List[Dict]:
        """
        Filter chunks by topic keywords.

        Args:
            topic_keywords: Keywords to search for

        Returns:
            Chunks that contain any of the keywords
        """
        relevant_chunks = []

        for chunk in self.chunks:
            chunk_text_lower = chunk['text'].lower()

            # Check if any keyword appears in the chunk
            for keyword in topic_keywords:
                if keyword.lower() in chunk_text_lower:
                    relevant_chunks.append(chunk)
                    break

        return relevant_chunks


def test_chunker():
    """Test the text chunker."""
    print("=" * 60)
    print("Testing Text Chunker")
    print("=" * 60)

    chunker = TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.process_all_texts()

    print("\n" + "=" * 60)
    print("Sample Chunks:")
    print("=" * 60)

    # Show first few chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i + 1}:")
        print(f"Source: {chunk['metadata']['source']}")
        print(f"Words: {chunk['metadata']['word_count']}")
        print(f"Text preview: {chunk['text'][:200]}...")

    # Test topic filtering
    print("\n" + "=" * 60)
    print("Testing Topic Filtering:")
    print("=" * 60)

    suffering_chunks = chunker.get_chunks_by_topic(['suffering', 'dukkha'])
    print(f"\nChunks mentioning 'suffering/dukkha': {len(suffering_chunks)}")

    impermanence_chunks = chunker.get_chunks_by_topic(['impermanence', 'anicca', 'change'])
    print(f"Chunks mentioning 'impermanence/anicca': {len(impermanence_chunks)}")

    return chunker


if __name__ == "__main__":
    test_chunker()
