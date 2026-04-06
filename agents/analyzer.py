"""
Data Analyzer Agent
Analyzes and synthesizes information from multiple sources
"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json


class DataAnalyzer:
    """Agent chuyên phân tích và tổng hợp dữ liệu từ nhiều nguồn"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.3,  # Lower temperature for more focused analysis
            api_key=api_key
        )
    
    def analyze_sources(self, sources: List[Dict], topic: str) -> Dict[str, Any]:
        """Phân tích độ tin cậy và chất lượng của các nguồn"""
        
        prompt = f"""Analyze the following sources about "{topic}":

Sources:
{self._format_sources_for_analysis(sources)}

Please provide:
1. Credibility Assessment (rate each source 1-10)
2. Source Bias Detection
3. Information Quality
4. Relevance to Topic
5. Conflicting Information

Return as JSON format:
{{
    "source_ratings": [],
    "bias_analysis": "",
    "quality_summary": "",
    "conflicts": [],
    "recommendations": []
}}
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert data analyst specializing in source evaluation and information synthesis."),
            HumanMessage(content=prompt)
        ])
        
        try:
            analysis = json.loads(response.content)
        except:
            analysis = {
                "source_ratings": [],
                "analysis": response.content
            }
        
        return analysis
    
    def extract_key_insights(self, documents: List[Dict], topic: str) -> Dict[str, Any]:
        """Trích xuất insights chính từ các documents"""
        
        prompt = f"""Extract key insights from these documents about "{topic}":

Documents:
{self._format_documents(documents)}

Provide:
1. Main Themes (3-5 themes)
2. Key Statistics and Data Points
3. Expert Opinions
4. Trends and Patterns
5. Important Facts

Format as structured JSON.
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert at extracting actionable insights from research data."),
            HumanMessage(content=prompt)
        ])
        
        return {
            "insights": response.content,
            "topic": topic,
            "num_documents": len(documents)
        }
    
    def identify_gaps(self, documents: List[Dict], topic: str) -> List[str]:
        """Xác định những khoảng trống trong thông tin"""
        
        prompt = f"""Based on these documents about "{topic}", identify:

Documents:
{self._format_documents(documents)}

What information is MISSING or INCOMPLETE?
What questions remain UNANSWERED?
What areas need MORE research?

List 5-10 specific gaps.
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are a critical analyst identifying gaps in research coverage."),
            HumanMessage(content=prompt)
        ])
        
        # Parse gaps from response
        gaps = [line.strip() for line in response.content.split('\n') if line.strip() and line[0].isdigit()]
        
        return gaps
    
    def synthesize_findings(self, 
                          research_data: str, 
                          analysis_results: Dict,
                          topic: str) -> str:
        """Tổng hợp tất cả findings thành một báo cáo phân tích"""
        
        prompt = f"""Synthesize the following research and analysis into a comprehensive analytical summary:

Topic: {topic}

Research Findings:
{research_data}

Source Analysis:
{json.dumps(analysis_results, indent=2)}

Create a synthesis that:
1. Integrates findings from all sources
2. Highlights consensus and disagreements
3. Presents a balanced view
4. Draws meaningful conclusions
5. Identifies implications

Write in a clear, analytical style.
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert research analyst creating comprehensive syntheses."),
            HumanMessage(content=prompt)
        ])
        
        return response.content
    
    def compare_perspectives(self, documents: List[Dict], topic: str) -> Dict:
        """So sánh các quan điểm khác nhau về topic"""
        
        prompt = f"""Compare different perspectives on "{topic}" from these sources:

{self._format_documents(documents)}

Identify:
1. Areas of Agreement
2. Points of Disagreement
3. Unique Perspectives
4. Evolution of Understanding

Present as a balanced comparison.
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are skilled at comparative analysis of multiple perspectives."),
            HumanMessage(content=prompt)
        ])
        
        return {
            "comparison": response.content,
            "num_sources": len(documents)
        }
    
    def _format_sources_for_analysis(self, sources: List[Dict]) -> str:
        """Format sources cho analysis prompt"""
        formatted = []
        for i, source in enumerate(sources, 1):
            if source.get('success'):
                formatted.append(f"""
Source {i}:
- Title: {source.get('title', 'N/A')}
- URL: {source.get('url', 'N/A')}
- Author: {source.get('authors', 'Unknown')}
- Date: {source.get('publish_date', 'Unknown')}
- Content Preview: {source.get('text', '')[:500]}...
""")
        return "\n".join(formatted)
    
    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents cho prompts"""
        formatted = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', doc.get('text', ''))
            formatted.append(f"""
Document {i}:
{content[:800]}...
Source: {doc.get('metadata', {}).get('source', doc.get('url', 'N/A'))}
""")
        return "\n".join(formatted)
    
    def generate_summary_statistics(self, documents: List[Dict]) -> Dict:
        """Tạo summary statistics về documents"""
        
        total_docs = len(documents)
        total_words = sum(len(doc.get('content', doc.get('text', '')).split()) for doc in documents)
        
        # Extract sources
        sources = set()
        for doc in documents:
            source = doc.get('metadata', {}).get('source') or doc.get('url', '')
            if source:
                sources.add(source)
        
        return {
            "total_documents": total_docs,
            "total_words": total_words,
            "average_words_per_doc": total_words // total_docs if total_docs > 0 else 0,
            "unique_sources": len(sources),
            "sources_list": list(sources)
        }
    

class DataAnalyzer:
    """Agent chuyên phân tích và tổng hợp dữ liệu từ nhiều nguồn"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.3,  # Lower temperature for more focused analysis
            api_key=api_key
        )
    
    def analyze_sources(self, sources: List[Dict], topic: str) -> Dict[str, Any]:
        """Phân tích độ tin cậy và chất lượng của các nguồn"""
        
        prompt = f"""Analyze the following sources about "{topic}":

Sources:
{self._format_sources_for_analysis(sources)}

Please provide:
1. Credibility Assessment (rate each source 1-10)
2. Source Bias Detection
3. Information Quality
4. Relevance to Topic
5. Conflicting Information

Return as JSON format:
{{
    "source_ratings": [],
    "bias_analysis": "",
    "quality_summary": "",
    "conflicts": [],
    "recommendations": []
}}
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert data analyst specializing in source evaluation and information synthesis."),
            HumanMessage(content=prompt)
        ])
        
        try:
            analysis = json.loads(response.content)
        except:
            analysis = {
                "source_ratings": [],
                "analysis": response.content
            }
        
        return analysis
    
    def extract_key_insights(self, documents: List[Dict], topic: str) -> Dict[str, Any]:
        """Trích xuất insights chính từ các documents"""
        
        prompt = f"""Extract key insights from these documents about "{topic}":

Documents:
{self._format_documents(documents)}

Provide:
1. Main Themes (3-5 themes)
2. Key Statistics and Data Points
3. Expert Opinions
4. Trends and Patterns
5. Important Facts

Format as structured JSON.
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert at extracting actionable insights from research data."),
            HumanMessage(content=prompt)
        ])
        
        return {
            "insights": response.content,
            "topic": topic,
            "num_documents": len(documents)
        }
    
    def identify_gaps(self, documents: List[Dict], topic: str) -> List[str]:
        """Xác định những khoảng trống trong thông tin"""
        
        prompt = f"""Based on these documents about "{topic}", identify:

Documents:
{self._format_documents(documents)}

What information is MISSING or INCOMPLETE?
What questions remain UNANSWERED?
What areas need MORE research?

List 5-10 specific gaps.
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are a critical analyst identifying gaps in research coverage."),
            HumanMessage(content=prompt)
        ])
        
        # Parse gaps from response
        gaps = [line.strip() for line in response.content.split('\n') if line.strip() and line[0].isdigit()]
        
        return gaps
    
    def synthesize_findings(self, 
                          research_data: str, 
                          analysis_results: Dict,
                          topic: str) -> str:
        """Tổng hợp tất cả findings thành một báo cáo phân tích"""
        
        prompt = f"""Synthesize the following research and analysis into a comprehensive analytical summary:

Topic: {topic}

Research Findings:
{research_data}

Source Analysis:
{json.dumps(analysis_results, indent=2)}

Create a synthesis that:
1. Integrates findings from all sources
2. Highlights consensus and disagreements
3. Presents a balanced view
4. Draws meaningful conclusions
5. Identifies implications

Write in a clear, analytical style.
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are an expert research analyst creating comprehensive syntheses."),
            HumanMessage(content=prompt)
        ])
        
        return response.content
    
    def compare_perspectives(self, documents: List[Dict], topic: str) -> Dict:
        """So sánh các quan điểm khác nhau về topic"""
        
        prompt = f"""Compare different perspectives on "{topic}" from these sources:

{self._format_documents(documents)}

Identify:
1. Areas of Agreement
2. Points of Disagreement
3. Unique Perspectives
4. Evolution of Understanding

Present as a balanced comparison.
"""
        
        response = self.llm.invoke([
            SystemMessage(content="You are skilled at comparative analysis of multiple perspectives."),
            HumanMessage(content=prompt)
        ])
        
        return {
            "comparison": response.content,
            "num_sources": len(documents)
        }
    
    def _format_sources_for_analysis(self, sources: List[Dict]) -> str:
        """Format sources cho analysis prompt"""
        formatted = []
        for i, source in enumerate(sources, 1):
            if source.get('success'):
                formatted.append(f"""
Source {i}:
- Title: {source.get('title', 'N/A')}
- URL: {source.get('url', 'N/A')}
- Author: {source.get('authors', 'Unknown')}
- Date: {source.get('publish_date', 'Unknown')}
- Content Preview: {source.get('text', '')[:500]}...
""")
        return "\n".join(formatted)
    
    def _format_documents(self, documents: List[Dict]) -> str:
        """Format documents cho prompts"""
        formatted = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', doc.get('text', ''))
            formatted.append(f"""
Document {i}:
{content[:800]}...
Source: {doc.get('metadata', {}).get('source', doc.get('url', 'N/A'))}
""")
        return "\n".join(formatted)
    
    def generate_summary_statistics(self, documents: List[Dict]) -> Dict:
        """Tạo summary statistics về documents"""
        
        total_docs = len(documents)
        total_words = sum(len(doc.get('content', doc.get('text', '')).split()) for doc in documents)
        
        # Extract sources
        sources = set()
        for doc in documents:
            source = doc.get('metadata', {}).get('source') or doc.get('url', '')
            if source:
                sources.add(source)
        
        return {
            "total_documents": total_docs,
            "total_words": total_words,
            "average_words_per_doc": total_words // total_docs if total_docs > 0 else 0,
            "unique_sources": len(sources),
            "sources_list": list(sources)
        }