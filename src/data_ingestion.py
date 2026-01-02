"""
Data Ingestion Module

This module handles reading blog files from various formats (Markdown, Text, Word)
and converting them into a structured JSONL format for further processing.

Key functions:
- read_blogs(): Main entry point, orchestrates the ingestion process
- parse_markdown(): Extract content from .md files
- parse_txt(): Extract content from .txt files
- parse_docx(): Extract content from .docx files
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import re

# Import file format parsers
import markdown
from docx import Document

logger = logging.getLogger(__name__)


def parse_markdown(file_path: Path) -> Dict[str, str]:
    """
    Parse a markdown file and extract its content.

    Args:
        file_path: Path to the .md file

    Returns:
        Dictionary with title, content, and metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try to extract title from first H1 heading
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem

        # Convert markdown to plain text (removing formatting)
        # This preserves readability while removing markdown syntax
        html = markdown.markdown(content)
        # Simple HTML tag removal (for plain text)
        plain_text = re.sub(r'<[^>]+>', '', html)

        return {
            'title': title,
            'content': plain_text.strip(),
            'raw_content': content,  # Keep original markdown
            'format': 'markdown'
        }

    except Exception as e:
        logger.error(f"Error parsing markdown file {file_path}: {e}")
        raise


def parse_txt(file_path: Path) -> Dict[str, str]:
    """
    Parse a plain text file.

    Args:
        file_path: Path to the .txt file

    Returns:
        Dictionary with title, content, and metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Use filename as title for plain text files
        # Or try to extract from first line if it looks like a title
        lines = content.split('\n')
        title = lines[0].strip() if lines and len(lines[0]) < 100 else file_path.stem

        return {
            'title': title,
            'content': content.strip(),
            'raw_content': content,
            'format': 'text'
        }

    except Exception as e:
        logger.error(f"Error parsing text file {file_path}: {e}")
        raise


def parse_docx(file_path: Path) -> Dict[str, str]:
    """
    Parse a Microsoft Word document.

    Args:
        file_path: Path to the .docx file

    Returns:
        Dictionary with title, content, and metadata
    """
    try:
        doc = Document(file_path)

        # Extract all paragraphs
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

        # Use first paragraph as title if it's short, otherwise use filename
        title = paragraphs[0] if paragraphs and len(paragraphs[0]) < 100 else file_path.stem

        # Join all paragraphs with newlines
        content = '\n\n'.join(paragraphs)

        return {
            'title': title,
            'content': content.strip(),
            'raw_content': content,
            'format': 'docx'
        }

    except Exception as e:
        logger.error(f"Error parsing Word document {file_path}: {e}")
        raise


def get_file_metadata(file_path: Path) -> Dict[str, any]:
    """
    Extract metadata from a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file metadata (size, dates, etc.)
    """
    stat = file_path.stat()

    return {
        'filename': file_path.name,
        'file_size_bytes': stat.st_size,
        'created_date': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified_date': datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def count_words(text: str) -> int:
    """
    Count words in a text string.

    Args:
        text: Input text

    Returns:
        Number of words
    """
    # Simple word count: split on whitespace and filter empty strings
    words = [word for word in text.split() if word]
    return len(words)


def read_blogs(
    input_dir: str,
    supported_formats: List[str] = ['md', 'txt', 'docx'],
    min_word_count: int = 100,
    max_word_count: int = 10000
) -> List[Dict]:
    """
    Read all blog files from a directory.

    This function:
    1. Scans the input directory for files with supported formats
    2. Parses each file using the appropriate parser
    3. Filters out blogs that don't meet word count requirements
    4. Adds metadata to each blog entry

    Args:
        input_dir: Directory containing blog files
        supported_formats: List of file extensions to process
        min_word_count: Minimum words required (skip shorter blogs)
        max_word_count: Maximum words allowed (skip longer blogs or chunk them)

    Returns:
        List of dictionaries, each representing a processed blog
    """
    input_path = Path(input_dir)
    blogs = []

    # Mapping of file extensions to parser functions
    parsers = {
        'md': parse_markdown,
        'txt': parse_txt,
        'docx': parse_docx
    }

    logger.info(f"Scanning directory: {input_dir}")

    # Iterate through all files in the directory
    for file_path in input_path.rglob('*'):
        # Skip directories
        if not file_path.is_file():
            continue

        # Get file extension (without the dot)
        extension = file_path.suffix[1:].lower()

        # Check if this file type is supported
        if extension not in supported_formats:
            logger.debug(f"Skipping unsupported file: {file_path.name}")
            continue

        try:
            logger.info(f"Processing: {file_path.name}")

            # Parse the file using the appropriate parser
            parser = parsers[extension]
            blog_data = parser(file_path)

            # Add file metadata
            metadata = get_file_metadata(file_path)
            blog_data.update(metadata)

            # Count words in content
            word_count = count_words(blog_data['content'])
            blog_data['word_count'] = word_count

            # Filter by word count
            if word_count < min_word_count:
                logger.warning(
                    f"Skipping {file_path.name}: too short ({word_count} words < {min_word_count} minimum)"
                )
                continue

            if word_count > max_word_count:
                logger.warning(
                    f"Skipping {file_path.name}: too long ({word_count} words > {max_word_count} maximum)"
                )
                # TODO: In future, could chunk long documents instead of skipping
                continue

            # Add to blogs list
            blogs.append(blog_data)
            logger.info(f"✓ Added {file_path.name} ({word_count} words)")

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")
            # Continue processing other files even if one fails
            continue

    logger.info(f"Successfully processed {len(blogs)} blog files")
    return blogs


def save_to_jsonl(blogs: List[Dict], output_path: str) -> None:
    """
    Save blogs to JSONL format (one JSON object per line).

    JSONL format is efficient for large datasets and easy to process line-by-line.

    Args:
        blogs: List of blog dictionaries
        output_path: Path where JSONL file will be saved
    """
    output_file = Path(output_path)

    # Create parent directories if they don't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for blog in blogs:
                # Write each blog as a single line JSON
                json_line = json.dumps(blog, ensure_ascii=False)
                f.write(json_line + '\n')

        logger.info(f"✓ Saved {len(blogs)} blogs to {output_path}")

    except Exception as e:
        logger.error(f"Error saving to JSONL: {e}")
        raise


def run_ingestion(
    input_dir: str,
    output_path: str,
    config: Dict
) -> int:
    """
    Main entry point for the ingestion module.

    This function orchestrates the complete ingestion process:
    1. Read blog files from input directory
    2. Parse and validate them
    3. Save to JSONL format

    Args:
        input_dir: Directory containing blog files
        output_path: Path for output JSONL file
        config: Configuration dictionary from pipeline_config.yaml

    Returns:
        Number of blogs successfully ingested
    """
    # Extract configuration parameters
    data_config = config.get('data', {})
    supported_formats = data_config.get('formats', ['md', 'txt', 'docx'])
    min_words = data_config.get('min_word_count', 100)
    max_words = data_config.get('max_word_count', 10000)

    logger.info("=" * 60)
    logger.info("Starting blog ingestion")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Supported formats: {', '.join(supported_formats)}")
    logger.info(f"Word count range: {min_words} - {max_words}")
    logger.info("")

    # Read and parse blogs
    blogs = read_blogs(
        input_dir=input_dir,
        supported_formats=supported_formats,
        min_word_count=min_words,
        max_word_count=max_words
    )

    if not blogs:
        logger.warning("No valid blog files found!")
        return 0

    # Save to JSONL
    save_to_jsonl(blogs, output_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Ingestion complete: {len(blogs)} blogs processed")
    logger.info("=" * 60)

    return len(blogs)


if __name__ == '__main__':
    # Example usage when running this module directly
    import yaml

    # Load config
    with open('pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Run ingestion
    run_ingestion(
        input_dir='./data/raw',
        output_path='./data/processed/raw_blogs.jsonl',
        config=config
    )
