#!/usr/bin/env python3
"""
Extract text from PDF files and save as .txt files for RAG system.
"""

import PyPDF2
import os
from pathlib import Path

def extract_pdf_text(pdf_path, output_path):
    """Extract text from a PDF file and save to a text file."""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            print(f"Processing {pdf_path.name}...")
            print(f"  Pages: {len(pdf_reader.pages)}")

            for page_num, page in enumerate(pdf_reader.pages):
                text += page.extract_text()
                if (page_num + 1) % 10 == 0:
                    print(f"  Processed {page_num + 1} pages...")

            # Clean up the text a bit
            text = text.replace('\n\n\n', '\n\n')  # Remove excessive newlines

            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)

            # Count words
            word_count = len(text.split())
            print(f"  ✓ Extracted {word_count:,} words")
            print(f"  ✓ Saved to: {output_path}")
            return word_count

    except Exception as e:
        print(f"  ✗ Error processing {pdf_path}: {e}")
        return 0

def main():
    texts_dir = Path("texts")
    pdf_files = list(texts_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in texts/ directory")
        return

    print(f"Found {len(pdf_files)} PDF file(s)\n")

    total_words = 0
    for pdf_path in pdf_files:
        output_path = pdf_path.with_suffix('.txt')
        words = extract_pdf_text(pdf_path, output_path)
        total_words += words
        print()

    print("=" * 60)
    print(f"TOTAL WORDS EXTRACTED: {total_words:,}")
    if total_words >= 10000:
        print("✓ Meets minimum requirement (10,000+ words)")
    else:
        print(f"✗ Need {10000 - total_words:,} more words")
    print("=" * 60)

if __name__ == "__main__":
    main()
