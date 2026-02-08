"""
Web Search Utilities for Research Agent.
Supports SerpAPI for real searches and a simulated mode for development.
"""

import os
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import html2text

from config import SERPAPI_API_KEY


def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using SerpAPI.
    
    Args:
        query: Search query string
        num_results: Number of results to return
    
    Returns:
        List of search results with 'title', 'url', 'snippet'
    """
    if not SERPAPI_API_KEY:
        print("SerpAPI key not found. Using simulated search.")
        return search_web_simulated(query, num_results)
    
    try:
        from serpapi import GoogleSearch
        
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": num_results,
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        search_results = []
        for result in results.get("organic_results", [])[:num_results]:
            search_results.append({
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
            })
        
        return search_results
    
    except Exception as e:
        print(f"SerpAPI search failed: {e}. Falling back to simulated search.")
        return search_web_simulated(query, num_results)


def search_web_simulated(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Simulated web search for development and testing.
    Returns mock results based on the query.
    
    Args:
        query: Search query string
        num_results: Number of results to return
    
    Returns:
        List of simulated search results
    """
    # Generate realistic-looking simulated results
    base_results = [
        {
            "title": f"Comprehensive Guide to {query}",
            "url": f"https://example.com/guide/{query.replace(' ', '-').lower()}",
            "snippet": f"This comprehensive guide covers all aspects of {query}, including key concepts, best practices, and real-world applications. Learn everything you need to know about this important topic.",
        },
        {
            "title": f"{query}: An In-Depth Analysis",
            "url": f"https://research.example.org/analysis/{query.replace(' ', '-').lower()}",
            "snippet": f"Our in-depth analysis of {query} reveals important insights and trends. We examine the latest research and provide actionable recommendations for practitioners.",
        },
        {
            "title": f"The Complete {query} Handbook",
            "url": f"https://handbook.example.com/{query.replace(' ', '-').lower()}",
            "snippet": f"Everything you need to know about {query} in one place. This handbook covers fundamentals, advanced topics, case studies, and future directions.",
        },
        {
            "title": f"Understanding {query}: Key Concepts and Applications",
            "url": f"https://education.example.net/learn/{query.replace(' ', '-').lower()}",
            "snippet": f"A beginner-friendly introduction to {query}. We break down complex concepts into easy-to-understand explanations with practical examples.",
        },
        {
            "title": f"{query} - Latest Research and Developments",
            "url": f"https://journal.example.edu/research/{query.replace(' ', '-').lower()}",
            "snippet": f"Stay up-to-date with the latest research on {query}. This article summarizes recent developments, breakthrough discoveries, and emerging trends.",
        },
        {
            "title": f"Practical Applications of {query}",
            "url": f"https://practical.example.io/apply/{query.replace(' ', '-').lower()}",
            "snippet": f"Discover how {query} is being applied in real-world scenarios. Case studies, success stories, and implementation guidelines included.",
        },
        {
            "title": f"{query} Best Practices and Guidelines",
            "url": f"https://bestpractices.example.org/{query.replace(' ', '-').lower()}",
            "snippet": f"Industry-standard best practices for {query}. Follow these guidelines to ensure optimal results and avoid common pitfalls.",
        },
    ]
    
    return base_results[:num_results]


def scrape_url(url: str, timeout: int = 10) -> Optional[str]:
    """
    Scrape content from a URL and convert to clean text.
    
    Args:
        url: URL to scrape
        timeout: Request timeout in seconds
    
    Returns:
        Cleaned text content or None if failed
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Convert to markdown-like text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0
        
        text = h.handle(str(soup))
        
        # Clean up extra whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned_text = '\n\n'.join(lines)
        
        return cleaned_text[:5000]  # Limit to 5000 chars
    
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None


def research_topic(topic: str, num_sources: int = 5) -> Dict[str, any]:
    """
    Conduct comprehensive research on a topic.
    
    Args:
        topic: Topic to research
        num_sources: Number of sources to gather
    
    Returns:
        Dictionary with search results and gathered content
    """
    # Search for information
    search_results = search_web(topic, num_sources)
    
    # Compile research data
    research_data = {
        "topic": topic,
        "sources": search_results,
        "source_count": len(search_results),
        "snippets": [result["snippet"] for result in search_results],
        "combined_snippets": "\n\n".join([
            f"Source: {r['title']}\n{r['snippet']}" 
            for r in search_results
        ]),
    }
    
    return research_data


if __name__ == "__main__":
    # Test the web search
    test_topic = "artificial intelligence in healthcare"
    print(f"Testing search for: {test_topic}")
    results = research_topic(test_topic)
    print(f"Found {results['source_count']} sources:")
    for source in results['sources']:
        print(f"  - {source['title']}")
