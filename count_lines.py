#!/usr/bin/env python3
"""
Count lines of code and documentation in the project.

Usage:
    python count_lines.py
    uv run python count_lines.py
"""

import os
from pathlib import Path
from collections import defaultdict

# Directories to ignore
IGNORE_DIRS = {
    '.venv',
    'venv',
    'node_modules',
    '.git',
    '__pycache__',
    'models',
    '.pytest_cache',
    'dist',
    'build',
    '.next',
}

# Code file extensions
CODE_EXTENSIONS = {
    '.py': 'Python',
    '.js': 'JavaScript',
    '.jsx': 'React JSX',
    '.ts': 'TypeScript',
    '.tsx': 'React TSX',
    '.css': 'CSS',
    '.html': 'HTML',
    '.sh': 'Shell',
}

# Documentation and config file extensions
DOC_EXTENSIONS = {
    '.md': 'Markdown',
    '.txt': 'Text',
    '.yaml': 'YAML',
    '.yml': 'YAML',
    '.toml': 'TOML',
    '.json': 'JSON',
    '.jsonl': 'JSON Lines',
    '.ipynb': 'Jupyter Notebook',
}


def should_ignore(path: Path) -> bool:
    """Check if path should be ignored."""
    parts = path.parts
    return any(ignore_dir in parts for ignore_dir in IGNORE_DIRS)


def count_lines(file_path: Path) -> int:
    """Count non-empty lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Count all lines (including empty for total count)
            return len(lines)
    except Exception:
        return 0


def count_non_empty_lines(file_path: Path) -> int:
    """Count non-empty, non-comment lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                    count += 1
            return count
    except Exception:
        return 0


def scan_directory(root_path: Path) -> tuple[dict, dict]:
    """Scan directory and count lines by category."""
    code_stats = defaultdict(lambda: {'files': 0, 'lines': 0, 'non_empty': 0})
    doc_stats = defaultdict(lambda: {'files': 0, 'lines': 0, 'non_empty': 0})

    for path in root_path.rglob('*'):
        if path.is_file() and not should_ignore(path):
            ext = path.suffix.lower()

            if ext in CODE_EXTENSIONS:
                lang = CODE_EXTENSIONS[ext]
                code_stats[lang]['files'] += 1
                code_stats[lang]['lines'] += count_lines(path)
                code_stats[lang]['non_empty'] += count_non_empty_lines(path)

            elif ext in DOC_EXTENSIONS:
                doc_type = DOC_EXTENSIONS[ext]
                doc_stats[doc_type]['files'] += 1
                doc_stats[doc_type]['lines'] += count_lines(path)
                doc_stats[doc_type]['non_empty'] += count_non_empty_lines(path)

    return dict(code_stats), dict(doc_stats)


def print_table(title: str, stats: dict) -> int:
    """Print formatted table and return total lines."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)
    print(f"{'Type':<20} {'Files':>8} {'Lines':>10} {'Non-Empty':>12}")
    print('-' * 60)

    total_files = 0
    total_lines = 0
    total_non_empty = 0

    # Sort by lines descending
    sorted_stats = sorted(stats.items(), key=lambda x: x[1]['lines'], reverse=True)

    for file_type, counts in sorted_stats:
        print(f"{file_type:<20} {counts['files']:>8} {counts['lines']:>10} {counts['non_empty']:>12}")
        total_files += counts['files']
        total_lines += counts['lines']
        total_non_empty += counts['non_empty']

    print('-' * 60)
    print(f"{'TOTAL':<20} {total_files:>8} {total_lines:>10} {total_non_empty:>12}")

    return total_lines


def main():
    """Main entry point."""
    root = Path(__file__).parent

    print("\n" + "=" * 60)
    print(" PROJECT LINE COUNT SUMMARY")
    print(" LLM Finetuning Blog Twin")
    print("=" * 60)

    code_stats, doc_stats = scan_directory(root)

    code_total = print_table("CODE FILES", code_stats)
    doc_total = print_table("DOCUMENTATION & CONFIG FILES", doc_stats)

    print("\n" + "=" * 60)
    print(" GRAND TOTAL")
    print("=" * 60)
    print(f" Code:          {code_total:>10} lines")
    print(f" Documentation: {doc_total:>10} lines")
    print(f" Combined:      {code_total + doc_total:>10} lines")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
