"""
Web Search Tool
Provides multi-source web search capabilities
"""

from typing import List, Dict, Optional
import requests
from duckduckgo_search import DDGS
import time
from config import Config


class WebSearchTool:
    """Tool for searching information across the web"""
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.ddg = DDGS()
    
    def search_web(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search the web using DuckDuckGo
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with title, link, and snippet
        """
        max_results = max_results or self.max_results
        
        try:
            results = []
            ddg_results = self.ddg.text(query, max_results=max_results)
            
            for result in ddg_results:
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'source': 'duckduckgo'
                })
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return self._fallback_search(query, max_results)
    
    def _fallback_search(self, query: str, max_results: int) -> List[Dict]:
        """Fallback search method if primary fails"""
        # This is a placeholder - you could implement Google Custom Search or other APIs
        print("Using fallback search method...")
        return []
    
    def search_multiple_sources(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """
        Search multiple queries
        
        Args:
            queries: List of search queries
            
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        for query in queries:
            print(f"Searching: {query}")
            results[query] = self.search_web(query)
            time.sleep(1)  # Rate limiting
        
        return results
    
    def search_news(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search for news articles
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of news results
        """
        max_results = max_results or self.max_results
        
        try:
            results = []
            news_results = self.ddg.news(query, max_results=max_results)
            
            for result in news_results:
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('url', ''),
                    'snippet': result.get('body', ''),
                    'date': result.get('date', ''),
                    'source': result.get('source', 'unknown'),
                    'type': 'news'
                })
            
            return results
            
        except Exception as e:
            print(f"News search error: {e}")
            return []
    
    def search_with_filters(self, 
                          query: str,
                          time_range: Optional[str] = None,
                          region: Optional[str] = None) -> List[Dict]:
        """
        Search with additional filters
        
        Args:
            query: Search query
            time_range: Time range filter (e.g., 'd' for day, 'w' for week)
            region: Region code (e.g., 'us-en')
            
        Returns:
            Filtered search results
        """
        try:
            results = []
            
            # Add time filter to query if specified
            if time_range:
                if time_range == 'd':
                    query += " (past 24 hours)"
                elif time_range == 'w':
                    query += " (past week)"
                elif time_range == 'm':
                    query += " (past month)"
            
            ddg_results = self.ddg.text(
                query,
                max_results=self.max_results,
                region=region or 'wt-wt'
            )
            
            for result in ddg_results:
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'source': 'duckduckgo'
                })
            
            return results
            
        except Exception as e:
            print(f"Filtered search error: {e}")
            return []
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """
        Get search suggestions for a query
        
        Args:
            query: Partial search query
            
        Returns:
            List of suggested queries
        """
        try:
            suggestions = self.ddg.suggestions(query)
            return [s.get('phrase', '') for s in suggestions if s.get('phrase')]
        except Exception as e:
            print(f"Suggestions error: {e}")
            return []
    
    def search_scholarly(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search for scholarly/academic content
        
        Args:
            query: Academic search query
            max_results: Maximum results
            
        Returns:
            Academic search results
        """
        # Add academic keywords to improve results
        academic_query = f"{query} research paper OR study OR journal"
        return self.search_web(academic_query, max_results)
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate results based on URL
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results
        """
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get('link', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results
    
    def rank_results_by_relevance(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Simple relevance ranking based on keyword matching
        
        Args:
            results: Search results
            query: Original query
            
        Returns:
            Ranked results
        """
        query_words = set(query.lower().split())
        
        def calculate_relevance(result: Dict) -> float:
            text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
            words = set(text.split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(words))
            return overlap / len(query_words) if query_words else 0
        
        # Sort by relevance score
        ranked = sorted(results, key=calculate_relevance, reverse=True)
        return ranked
    
    def search_and_rank(self, query: str, max_results: Optional[int] = None) -> List[Dict]:
        """
        Search and return ranked results
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            Ranked search results
        """
        results = self.search_web(query, max_results)
        results = self.deduplicate_results(results)
        results = self.rank_results_by_relevance(results, query)
        return results


class GoogleSearchTool:
    """Google Custom Search API wrapper (optional, requires API key)"""
    
    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search using Google Custom Search API"""
        if not self.api_key or not self.cse_id:
            return []
        
        try:
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': query,
                'num': min(num_results, 10)  # Max 10 per request
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'source': 'google'
                })
            
            return results
            
        except Exception as e:
            print(f"Google search error: {e}")
            return []