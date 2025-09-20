#!/usr/bin/env python3
"""
Improved PDF Paragraph Splitter for Legal Documents

This script splits a PDF file into chunks based on paragraphs, sections, and logical divisions
with configurable overlap. Optimized for legal documents and structured text.
"""

import pdfplumber
import re
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
import argparse


class ImprovedPDFSplitter:
    """Improved class to handle PDF paragraph splitting with better legal document support."""
    
    def __init__(self, overlap_sentences: int = 3, min_chunk_size: int = 200, max_chunk_size: int = 2000):
        """
        Initialize the PDF splitter.
        
        Args:
            overlap_sentences: Number of sentences to overlap between chunks
            min_chunk_size: Minimum character count for a chunk
            max_chunk_size: Maximum character count for a chunk
        """
        self.overlap_sentences = overlap_sentences
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF file."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        print(f"Processed page {page_num + 1}")
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers that appear alone
        text = re.sub(r'\n\d+\n', '\n', text)
        # Clean up line breaks
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def find_section_headers(self, text: str) -> List[Tuple[int, str]]:
        """Find section headers and their positions."""
        # Common legal document section patterns
        section_patterns = [
            r'^[A-Z][A-Z\.\s]+:$',  # ALL CAPS headers
            r'^\d+\.\s+[A-Z]',      # Numbered sections
            r'^[A-Z]\.\s+[A-Z]',    # Lettered sections (A. B. C.)
            r'^[IVX]+\.\s+[A-Z]',   # Roman numeral sections
            r'^[a-z]\.\s+[A-Z]',    # Lowercase letter sections
            r'^\d+\.\d+\s+[A-Z]',   # Sub-sections
        ]
        
        headers = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) > 3:  # Ignore very short lines
                for pattern in section_patterns:
                    if re.match(pattern, line):
                        headers.append((i, line))
                        break
        
        return headers
    
    def split_by_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split text by sections and headers."""
        headers = self.find_section_headers(text)
        lines = text.split('\n')
        sections = []
        
        for i, (header_line_num, header_text) in enumerate(headers):
            start_line = header_line_num
            end_line = headers[i + 1][0] if i + 1 < len(headers) else len(lines)
            
            section_content = '\n'.join(lines[start_line:end_line]).strip()
            
            if len(section_content) >= self.min_chunk_size:
                sections.append({
                    'type': 'section',
                    'header': header_text,
                    'content': section_content,
                    'start_line': start_line,
                    'end_line': end_line,
                    'char_count': len(section_content)
                })
        
        return sections
    
    def split_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Split text into paragraphs."""
        # Clean up the text first
        text = self.clean_text(text)
        
        # Split by double newlines or clear paragraph markers
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Also split by sentence patterns that indicate new topics
        additional_splits = []
        for para in paragraphs:
            # Split on patterns like "The Hon'ble Justice", case citations, etc.
            if re.search(r'For the|The Hon\'ble|MAT No\.|APPEAL No\.', para):
                parts = re.split(r'(?=For the|The Hon\'ble|MAT No\.|APPEAL No\.)', para)
                if len(parts) > 1:
                    additional_splits.extend([p.strip() for p in parts if p.strip()])
                else:
                    additional_splits.append(para)
            else:
                additional_splits.append(para)
        
        paragraphs = additional_splits
        
        # Filter and clean paragraphs
        filtered_paragraphs = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para and len(para) > self.min_chunk_size:
                filtered_paragraphs.append({
                    'type': 'paragraph',
                    'content': para,
                    'paragraph_index': i,
                    'char_count': len(para)
                })
        
        return filtered_paragraphs
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better legal document handling."""
        # Enhanced sentence splitting for legal documents
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=\.)\s+(?=For the)|(?<=\.)\s+(?=The Hon\'ble)'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return sentences
    
    def create_smart_chunks(self, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks with intelligent overlap and size management."""
        chunks = []
        
        for item in content_items:
            content = item['content']
            sentences = self.split_into_sentences(content)
            
            if len(content) <= self.max_chunk_size:
                # Content fits in one chunk
                chunk = {
                    'chunk_id': len(chunks) + 1,
                    'content': content,
                    'type': item['type'],
                    'header': item.get('header', ''),
                    'char_count': len(content),
                    'sentence_count': len(sentences),
                    'overlap_sentences': 0
                }
                chunks.append(chunk)
            else:
                # Split into multiple chunks with overlap
                chunk_sentences = []
                start_idx = 0
                
                while start_idx < len(sentences):
                    # Calculate how many sentences to include
                    end_idx = start_idx + 10  # Start with 10 sentences
                    
                    # Adjust to stay within max_chunk_size
                    while end_idx <= len(sentences):
                        chunk_text = ' '.join(sentences[start_idx:end_idx])
                        if len(chunk_text) > self.max_chunk_size and end_idx > start_idx + 5:
                            end_idx -= 1
                            break
                        end_idx += 1
                    
                    if end_idx > len(sentences):
                        end_idx = len(sentences)
                    
                    chunk_sentences = sentences[start_idx:end_idx]
                    chunk_content = ' '.join(chunk_sentences)
                    
                    # Add overlap from previous chunk if applicable
                    overlap_sentences = []
                    if start_idx > 0 and self.overlap_sentences > 0:
                        overlap_start = max(0, start_idx - self.overlap_sentences)
                        overlap_sentences = sentences[overlap_start:start_idx]
                        chunk_content = ' '.join(overlap_sentences + chunk_sentences)
                    
                    if len(chunk_content) >= self.min_chunk_size:
                        chunk = {
                            'chunk_id': len(chunks) + 1,
                            'content': chunk_content,
                            'type': item['type'],
                            'header': item.get('header', ''),
                            'char_count': len(chunk_content),
                            'sentence_count': len(chunk_sentences),
                            'overlap_sentences': len(overlap_sentences)
                        }
                        chunks.append(chunk)
                    
                    # Move start index with some overlap
                    start_idx += max(5, len(chunk_sentences) - self.overlap_sentences)
        
        return chunks
    
    def split_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Main method to split PDF into chunks."""
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print("No text extracted from PDF")
            return []
        
        print(f"Extracted {len(text)} characters from PDF")
        
        # Try to split by sections first
        sections = self.split_by_sections(text)
        print(f"Found {len(sections)} sections")
        
        if len(sections) > 1:
            # Use sections as primary division
            chunks = self.create_smart_chunks(sections)
        else:
            # Fall back to paragraph splitting
            paragraphs = self.split_by_paragraphs(text)
            print(f"Found {len(paragraphs)} paragraphs")
            chunks = self.create_smart_chunks(paragraphs)
        
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_path: str, format: str = 'json'):
        """Save chunks to file."""
        if format.lower() == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
        elif format.lower() == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(f"=== CHUNK {chunk['chunk_id']} ===\n")
                    f.write(f"Type: {chunk['type']}\n")
                    if chunk.get('header'):
                        f.write(f"Header: {chunk['header']}\n")
                    f.write(f"Sentences: {chunk['sentence_count']}\n")
                    f.write(f"Characters: {chunk['char_count']}\n")
                    f.write(f"Overlap: {chunk.get('overlap_sentences', 0)}\n")
                    f.write("\n")
                    f.write(chunk['content'])
                    f.write("\n\n" + "="*50 + "\n\n")
        
        print(f"Chunks saved to: {output_path}")


def main():
    """Main function to run the improved PDF splitter."""
    parser = argparse.ArgumentParser(description='Split PDF into intelligent chunks with overlap')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', default='pdf_chunks_improved.json', help='Output file path')
    parser.add_argument('--format', '-f', choices=['json', 'txt'], default='json', help='Output format')
    parser.add_argument('--overlap', type=int, default=3, help='Number of sentences to overlap')
    parser.add_argument('--min-size', type=int, default=200, help='Minimum chunk size in characters')
    parser.add_argument('--max-size', type=int, default=2000, help='Maximum chunk size in characters')
    
    args = parser.parse_args()
    
    # Create splitter instance
    splitter = ImprovedPDFSplitter(
        overlap_sentences=args.overlap,
        min_chunk_size=args.min_size,
        max_chunk_size=args.max_size
    )
    
    # Split PDF
    chunks = splitter.split_pdf(args.pdf_path)
    
    if chunks:
        # Save chunks
        splitter.save_chunks(chunks, args.output, args.format)
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total chunks: {len(chunks)}")
        print(f"Average chunk size: {sum(c['char_count'] for c in chunks) / len(chunks):.1f} characters")
        print(f"Chunks with overlap: {sum(1 for c in chunks if c.get('overlap_sentences', 0) > 0)}")
        print(f"Section chunks: {sum(1 for c in chunks if c['type'] == 'section')}")
        print(f"Paragraph chunks: {sum(1 for c in chunks if c['type'] == 'paragraph')}")
    else:
        print("No chunks created")


if __name__ == "__main__":
    main()
