"""
Helper utilities for the research assistant
"""

import re
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib


def format_timestamp(timestamp: Optional[float] = None) -> str:
    """Format timestamp thành string dễ đọc"""
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def clean_text(text: str) -> str:
    """Làm sạch text, loại bỏ ký tự đặc biệt và khoảng trắng thừa"""
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', '', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """Trích xuất keywords từ text (simple implementation)"""
    # Remove common stop words (simplified list)
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
        'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this',
        'it', 'from', 'be', 'are', 'was', 'were', 'been', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
    }
    
    # Clean and tokenize
    text = clean_text(text.lower())
    words = re.findall(r'\b[a-z]{3,}\b', text)
    
    # Filter stop words and count frequency
    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top N
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:num_keywords]]


def calculate_similarity(text1: str, text2: str) -> float:
    """Tính similarity đơn giản giữa 2 texts (Jaccard similarity)"""
    # Tokenize
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def save_report(report: str, 
                topic: str, 
                output_dir: str = "./reports",
                format: str = "md") -> str:
    """Lưu report ra file"""
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    filename = generate_filename(topic, format)
    filepath = os.path.join(output_dir, filename)
    
    # Save file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return filepath


def load_report(filepath: str) -> str:
    """Đọc report từ file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def generate_filename(topic: str, extension: str = "md") -> str:
    """Generate unique filename từ topic"""
    # Clean topic for filename
    clean_topic = re.sub(r'[^\w\s-]', '', topic.lower())
    clean_topic = re.sub(r'[-\s]+', '_', clean_topic)
    
    # Truncate if too long
    if len(clean_topic) > 50:
        clean_topic = clean_topic[:50]
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"research_{clean_topic}_{timestamp}.{extension}"


def save_json(data: Dict, filepath: str) -> None:
    """Lưu data ra JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict:
    """Đọc JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def hash_text(text: str) -> str:
    """Tạo hash cho text"""
    return hashlib.md5(text.encode()).hexdigest()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chia text thành chunks với overlap"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks


def extract_urls(text: str) -> List[str]:
    """Trích xuất tất cả URLs từ text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return list(set(urls))  # Remove duplicates


def count_words(text: str) -> int:
    """Đếm số words trong text"""
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Ước tính reading time (minutes)"""
    word_count = count_words(text)
    minutes = word_count / words_per_minute
    return max(1, round(minutes))


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text đến max_length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def validate_url(url: str) -> bool:
    """Validate URL format"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None


def create_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Tạo markdown table"""
    # Header
    table = "| " + " | ".join(headers) + " |\n"
    
    # Separator
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Rows
    for row in rows:
        table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    
    return table


def format_file_size(size_bytes: int) -> str:
    """Format file size thành human readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_domain_from_url(url: str) -> str:
    """Extract domain từ URL"""
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if match:
        return match.group(1)
    return url


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def flatten_list(nested_list: List[Any]) -> List:
    """Flatten nested list"""
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_list(item))
        else:
            flat.append(item)
    return flat


def deduplicate_list(items: List[Any]) -> List[Any]:
    """Remove duplicates while preserving order"""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def safe_filename(filename: str) -> str:
    """Make filename safe for filesystem"""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    return filename


class ProgressTracker:
    """Simple progress tracker"""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
    
    def update(self, amount: int = 1):
        """Update progress"""
        self.current += amount
        percentage = (self.current / self.total) * 100
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%)", end='')
    
    def finish(self):
        """Mark as finished"""
        print()  # New line


class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"{self.name} took {duration:.2f} seconds")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0