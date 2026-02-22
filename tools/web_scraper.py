"""
Web Scraper Tool
Extracts content from web pages
"""

import requests
from bs4 import BeautifulSoup
from newspaper import Article
from typing import Dict, List, Optional
import time
from urllib.parse import urlparse
from config import Config


class WebScraper:
    """Tool for scraping content from web pages"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            'User-Agent': Config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.session = requests.Session()
    
    def scrape_url(self, url: str) -> Dict:
        """
        Scrape content from a URL using newspaper3k
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with scraped content
        """
        try:
            # Try newspaper3k first (best for articles)
            article = Article(url)
            article.download()
            article.parse()
            
            return {
                'url': url,
                'title': article.title,
                'text': article.text,
                'authors': article.authors,
                'publish_date': article.publish_date,
                'top_image': article.top_image,
                'images': list(article.images),
                'movies': article.movies,
                'keywords': article.keywords if hasattr(article, 'keywords') else [],
                'summary': article.summary if hasattr(article, 'summary') else '',
                'success': True,
                'method': 'newspaper'
            }
            
        except Exception as e:
            print(f"Newspaper3k failed for {url}: {e}")
            return self._fallback_scrape(url)
    
    def _fallback_scrape(self, url: str) -> Dict:
        """
        Fallback scraping method using BeautifulSoup
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with scraped content
        """
        try:
            response = self.session.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe']):
                element.decompose()
            
            # Extract title
            title = ''
            if soup.title:
                title = soup.title.string
            elif soup.find('h1'):
                title = soup.find('h1').get_text()
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Clean text
            text = self._clean_text(main_content)
            
            return {
                'url': url,
                'title': title.strip() if title else 'No Title',
                'text': text,
                'authors': [],
                'publish_date': None,
                'top_image': self._extract_main_image(soup),
                'success': True,
                'method': 'beautifulsoup'
            }
            
        except Exception as e:
            print(f"Fallback scraping failed for {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'success': False
            }
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from soup"""
        # Try common content containers
        content_tags = [
            soup.find('article'),
            soup.find('main'),
            soup.find('div', class_='content'),
            soup.find('div', class_='post-content'),
            soup.find('div', id='content'),
            soup.find('body')
        ]
        
        for tag in content_tags:
            if tag:
                return tag.get_text()
        
        return soup.get_text()
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Split into lines
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_main_image(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract main image URL"""
        # Try Open Graph image
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return og_image['content']
        
        # Try first image in article
        article = soup.find('article')
        if article:
            img = article.find('img')
            if img and img.get('src'):
                return img['src']
        
        # Try any image
        img = soup.find('img')
        if img and img.get('src'):
            return img['src']
        
        return None
    
    def scrape_multiple(self, urls: List[str], delay: float = 1.0) -> List[Dict]:
        """
        Scrape multiple URLs with delay
        
        Args:
            urls: List of URLs to scrape
            delay: Delay between requests in seconds
            
        Returns:
            List of scraped results
        """
        results = []
        
        for i, url in enumerate(urls):
            print(f"Scraping {i+1}/{len(urls)}: {url}")
            result = self.scrape_url(url)
            results.append(result)
            
            # Add delay between requests (except for last one)
            if i < len(urls) - 1:
                time.sleep(delay)
        
        return results
    
    def scrape_with_retry(self, url: str, max_retries: int = 3) -> Dict:
        """
        Scrape with retry logic
        
        Args:
            url: URL to scrape
            max_retries: Maximum number of retry attempts
            
        Returns:
            Scraped content
        """
        for attempt in range(max_retries):
            try:
                result = self.scrape_url(url)
                if result.get('success'):
                    return result
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        
        return {
            'url': url,
            'error': 'Max retries exceeded',
            'success': False
        }
    
    def get_page_metadata(self, url: str) -> Dict:
        """
        Extract metadata from a page
        
        Args:
            url: URL to extract metadata from
            
        Returns:
            Dictionary of metadata
        """
        try:
            response = self.session.get(url, headers=self.headers, timeout=self.timeout)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            metadata = {
                'url': url,
                'title': '',
                'description': '',
                'keywords': '',
                'author': '',
                'og_title': '',
                'og_description': '',
                'og_image': ''
            }
            
            # Title
            if soup.title:
                metadata['title'] = soup.title.string
            
            # Meta tags
            meta_tags = {
                'description': soup.find('meta', attrs={'name': 'description'}),
                'keywords': soup.find('meta', attrs={'name': 'keywords'}),
                'author': soup.find('meta', attrs={'name': 'author'}),
                'og_title': soup.find('meta', property='og:title'),
                'og_description': soup.find('meta', property='og:description'),
                'og_image': soup.find('meta', property='og:image')
            }
            
            for key, tag in meta_tags.items():
                if tag:
                    content = tag.get('content') or tag.get('value')
                    if content:
                        metadata[key] = content
            
            return metadata
            
        except Exception as e:
            print(f"Metadata extraction error: {e}")
            return {'url': url, 'error': str(e)}
    
    def is_scrapable(self, url: str) -> bool:
        """
        Check if URL is scrapable
        
        Args:
            url: URL to check
            
        Returns:
            True if likely scrapable, False otherwise
        """
        try:
            # Check URL format
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Try HEAD request
            response = self.session.head(url, headers=self.headers, timeout=5, allow_redirects=True)
            
            # Check status code
            if response.status_code >= 400:
                return False
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                return False
            
            return True
            
        except Exception as e:
            print(f"Scrapability check failed: {e}")
            return False
    
    def extract_links(self, url: str, internal_only: bool = False) -> List[str]:
        """
        Extract all links from a page
        
        Args:
            url: URL to extract links from
            internal_only: Only return internal links
            
        Returns:
            List of URLs
        """
        try:
            response = self.session.get(url, headers=self.headers, timeout=self.timeout)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            base_domain = urlparse(url).netloc
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Skip empty or anchor-only links
                if not href or href.startswith('#'):
                    continue
                
                # Convert relative to absolute
                if href.startswith('/'):
                    href = f"{urlparse(url).scheme}://{base_domain}{href}"
                elif not href.startswith('http'):
                    continue
                
                # Filter internal links if requested
                if internal_only:
                    if urlparse(href).netloc == base_domain:
                        links.append(href)
                else:
                    links.append(href)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            print(f"Link extraction error: {e}")
            return []
    
    def batch_scrape(self, urls: List[str], batch_size: int = 5) -> List[Dict]:
        """
        Scrape URLs in batches
        
        Args:
            urls: List of URLs
            batch_size: Number of URLs per batch
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(urls)-1)//batch_size + 1}")
            
            batch_results = self.scrape_multiple(batch, delay=1.0)
            results.extend(batch_results)
            
            # Pause between batches
            if i + batch_size < len(urls):
                print("Pausing between batches...")
                time.sleep(3)
        
        return results