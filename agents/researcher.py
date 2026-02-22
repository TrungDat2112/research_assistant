"""
Research Agent - Main orchestrator for research tasks
Coordinates searching, scraping, analysis, and report writing
"""

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from tools.web_search import WebSearchTool
from tools.web_scraper import WebScraper
from tools.vector_store import VectorStore
from agents.analyzer import DataAnalyzer
from agents.writer import ReportWriter
from typing import List, Dict, Optional
from config import Config
import time


class ResearchAgent:
    """Main research orchestrator using CrewAI framework"""
    
    def __init__(self, api_key: str):
        """
        Initialize Research Agent
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=Config.RESEARCHER_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=api_key
        )
        
        # Initialize tools
        self.search_tool = WebSearchTool(max_results=Config.MAX_SEARCH_RESULTS)
        self.scraper = WebScraper()
        self.vector_store = VectorStore()
        
        # Initialize specialized agents
        self.analyzer = DataAnalyzer(api_key=api_key, model=Config.ANALYZER_MODEL)
        self.writer = ReportWriter(api_key=api_key, model=Config.WRITER_MODEL)
        
        # Define CrewAI agents
        self._setup_crew_agents()
    
    def _setup_crew_agents(self):
        """Setup CrewAI agents"""
        
        self.researcher_agent = Agent(
            role='Senior Research Specialist',
            goal='Conduct comprehensive research on any given topic by gathering information from multiple reliable sources',
            backstory="""You are a seasoned researcher with 15 years of experience in 
            academic and industry research. You excel at finding relevant, credible sources
            and extracting key information. You understand how to evaluate source quality
            and identify authoritative references.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.analyst_agent = Agent(
            role='Data Analysis Expert',
            goal='Analyze research data, identify patterns, and synthesize insights from multiple sources',
            backstory="""You are an expert data analyst with a PhD in Information Science.
            You specialize in qualitative data analysis, pattern recognition, and synthesis
            of complex information from diverse sources. You can spot inconsistencies,
            identify trends, and draw meaningful conclusions.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        self.writer_agent = Agent(
            role='Professional Technical Writer',
            goal='Create clear, comprehensive, and well-structured research reports',
            backstory="""You are an award-winning technical writer with expertise in
            creating research reports, white papers, and academic publications. You know
            how to organize complex information logically, write clearly for diverse
            audiences, and properly cite sources.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def research_topic(self, 
                      topic: str,
                      num_sources: int = 5,
                      include_news: bool = False,
                      depth: str = "standard") -> Dict:
        """
        Conduct comprehensive research on a topic
        
        Args:
            topic: Research topic
            num_sources: Number of sources to analyze
            include_news: Include news sources
            depth: Research depth ('quick', 'standard', 'deep')
            
        Returns:
            Dictionary with research results
        """
        print(f"\n{'='*70}")
        print(f"🔬 Starting Research: {topic}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Adjust parameters based on depth
        depth_params = self._get_depth_parameters(depth, num_sources)
        
        # Phase 1: Search
        print("📊 Phase 1: Web Search")
        print("-" * 70)
        search_results = self._search_phase(topic, depth_params, include_news)
        
        # Phase 2: Scraping
        print("\n📄 Phase 2: Content Extraction")
        print("-" * 70)
        scraped_data = self._scraping_phase(search_results, depth_params['max_scrape'])
        
        # Phase 3: Vector Storage
        print("\n💾 Phase 3: Building Knowledge Base")
        print("-" * 70)
        self._storage_phase(scraped_data)
        
        # Phase 4: Information Retrieval
        print("\n🔍 Phase 4: Information Retrieval")
        print("-" * 70)
        relevant_docs = self.vector_store.similarity_search(topic, k=depth_params['retrieval_k'])
        
        # Phase 5: Analysis
        print("\n🧠 Phase 5: Analysis & Synthesis")
        print("-" * 70)
        analysis_results = self._analysis_phase(scraped_data, relevant_docs, topic)
        
        # Phase 6: Report Generation
        print("\n✍️  Phase 6: Report Generation")
        print("-" * 70)
        report = self._writing_phase(topic, scraped_data, analysis_results, relevant_docs)
        
        # Calculate metrics
        elapsed_time = time.time() - start_time
        
        result = {
            'topic': topic,
            'report': report,
            'sources': scraped_data,
            'num_sources': len([s for s in scraped_data if s.get('success')]),
            'analysis': analysis_results,
            'search_results': search_results,
            'elapsed_time': elapsed_time,
            'depth': depth
        }
        
        print(f"\n{'='*70}")
        print(f"✅ Research Complete! ({elapsed_time:.1f}s)")
        print(f"{'='*70}\n")
        
        return result
    
    def _get_depth_parameters(self, depth: str, num_sources: int) -> Dict:
        """Get parameters based on research depth"""
        params = {
            'quick': {
                'search_queries': 1,
                'max_scrape': min(num_sources, 3),
                'retrieval_k': 5,
                'analysis_depth': 'basic'
            },
            'standard': {
                'search_queries': 2,
                'max_scrape': min(num_sources, 5),
                'retrieval_k': 10,
                'analysis_depth': 'standard'
            },
            'deep': {
                'search_queries': 3,
                'max_scrape': min(num_sources, 10),
                'retrieval_k': 15,
                'analysis_depth': 'comprehensive'
            }
        }
        
        return params.get(depth, params['standard'])
    
    def _search_phase(self, topic: str, params: Dict, include_news: bool) -> List[Dict]:
        """Execute search phase"""
        all_results = []
        
        # Main search
        print(f"Searching for: {topic}")
        results = self.search_tool.search_web(topic)
        all_results.extend(results)
        
        # Additional queries for deeper research
        if params['search_queries'] > 1:
            additional_queries = self._generate_related_queries(topic, params['search_queries'] - 1)
            for query in additional_queries:
                print(f"Additional search: {query}")
                results = self.search_tool.search_web(query)
                all_results.extend(results)
        
        # News search if requested
        if include_news:
            print(f"Searching news: {topic}")
            news_results = self.search_tool.search_news(topic)
            all_results.extend(news_results)
        
        # Deduplicate
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get('link', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        print(f"✓ Found {len(unique_results)} unique sources")
        return unique_results
    
    def _generate_related_queries(self, topic: str, num_queries: int) -> List[str]:
        """Generate related search queries"""
        variations = [
            f"{topic} overview",
            f"{topic} latest research",
            f"{topic} expert analysis",
            f"what is {topic}",
            f"{topic} trends",
            f"{topic} statistics"
        ]
        return variations[:num_queries]
    
    def _scraping_phase(self, search_results: List[Dict], max_scrape: int) -> List[Dict]:
        """Execute scraping phase"""
        # Get top URLs
        urls_to_scrape = [r['link'] for r in search_results[:max_scrape] if r.get('link')]
        
        # Scrape with progress
        scraped_data = []
        for i, url in enumerate(urls_to_scrape, 1):
            print(f"[{i}/{len(urls_to_scrape)}] Scraping: {url[:60]}...")
            result = self.scraper.scrape_url(url)
            scraped_data.append(result)
            time.sleep(1)  # Rate limiting
        
        successful = len([s for s in scraped_data if s.get('success')])
        print(f"✓ Successfully scraped {successful}/{len(urls_to_scrape)} pages")
        
        return scraped_data
    
    def _storage_phase(self, scraped_data: List[Dict]):
        """Execute storage phase"""
        num_chunks = self.vector_store.add_documents(scraped_data)
        stats = self.vector_store.get_collection_stats()
        
        print(f"✓ Stored {num_chunks} chunks")
        print(f"✓ Total chunks in DB: {stats.get('total_chunks', 0)}")
    
    def _analysis_phase(self, scraped_data: List[Dict], relevant_docs: List[Dict], topic: str) -> Dict:
        """Execute analysis phase"""
        
        # Source analysis
        print("Analyzing source credibility...")
        source_analysis = self.analyzer.analyze_sources(scraped_data, topic)
        
        # Extract insights
        print("Extracting key insights...")
        insights = self.analyzer.extract_key_insights(relevant_docs, topic)
        
        # Identify gaps
        print("Identifying information gaps...")
        gaps = self.analyzer.identify_gaps(relevant_docs, topic)
        
        # Generate statistics
        stats = self.analyzer.generate_summary_statistics(relevant_docs)
        
        print(f"✓ Analysis complete")
        
        return {
            'source_analysis': source_analysis,
            'insights': insights,
            'gaps': gaps,
            'statistics': stats
        }
    
    def _writing_phase(self, 
                       topic: str,
                       scraped_data: List[Dict],
                       analysis: Dict,
                       relevant_docs: List[Dict]) -> str:
        """Execute writing phase using CrewAI"""
        
        # Prepare context
        sources_summary = self._format_sources(scraped_data)
        docs_summary = self._format_documents(relevant_docs)
        
        # Create tasks
        research_task = Task(
            description=f"""Compile research findings on: {topic}
            
Available Sources:
{sources_summary}

Document Database:
{docs_summary[:2000]}

Task: Create a comprehensive summary of findings including:
- Key facts and statistics
- Main concepts and definitions  
- Current state of knowledge
- Important developments
- Expert perspectives

Focus on accuracy and completeness.""",
            agent=self.researcher_agent,
            expected_output="Detailed research summary with key findings"
        )
        
        analysis_task = Task(
            description=f"""Analyze the research on: {topic}
            
Source Analysis Results:
{str(analysis.get('source_analysis', {}))[:1000]}

Key Insights:
{str(analysis.get('insights', {}))[:1000]}

Task: Provide analytical synthesis including:
- Pattern identification
- Consensus and disagreements
- Quality assessment
- Knowledge gaps
- Critical insights

Be thorough and objective.""",
            agent=self.analyst_agent,
            expected_output="Comprehensive analytical synthesis"
        )
        
        writing_task = Task(
            description=f"""Create a professional research report on: {topic}
            
Use the research findings and analysis to create a complete report with:

1. Executive Summary (150-200 words)
2. Introduction with context and objectives
3. Main Findings (well-organized sections)
4. Analysis & Insights
5. Conclusions and implications
6. References (properly formatted)

Requirements:
- Professional academic style
- Clear structure with headers
- Proper citations
- Approximately 1500-2000 words
- Markdown format

Make it comprehensive, well-written, and properly sourced.""",
            agent=self.writer_agent,
            expected_output="Complete professional research report in markdown"
        )
        
        # Execute crew
        print("Generating report with AI crew...")
        crew = Crew(
            agents=[self.researcher_agent, self.analyst_agent, self.writer_agent],
            tasks=[research_task, analysis_task, writing_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = crew.kickoff()
        
        print("✓ Report generated")
        
        return str(result)
    
    def _format_sources(self, sources: List[Dict]) -> str:
        """Format sources for prompts"""
        formatted = []
        for i, source in enumerate(sources, 1):
            if source.get('success'):
                formatted.append(
                    f"{i}. {source.get('title', 'N/A')}\n"
                    f"   URL: {source.get('url', 'N/A')}\n"
                    f"   Preview: {source.get('text', '')[:200]}..."
                )
        return "\n\n".join(formatted[:10])  # Limit to prevent token overflow
    
    def _format_documents(self, docs: List[Dict]) -> str:
        """Format documents for prompts"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.get('content', '')
            source = doc.get('metadata', {}).get('source', 'N/A')
            formatted.append(f"{i}. {content[:400]}...\n   Source: {source}")
        return "\n\n".join(formatted[:10])
    
    def quick_research(self, topic: str) -> str:
        """Quick research with minimal depth"""
        result = self.research_topic(topic, num_sources=3, depth='quick')
        return result['report']
    
    def deep_research(self, topic: str) -> Dict:
        """Deep research with maximum depth"""
        return self.research_topic(topic, num_sources=10, depth='deep', include_news=True)