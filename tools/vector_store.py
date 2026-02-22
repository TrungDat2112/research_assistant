"""
Vector Store Tool
Manages document storage and retrieval using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional, Any
import os
from config import Config


class VectorStore:
    """Manages vector storage for research documents"""
    
    def __init__(self, 
                 collection_name: str = None,
                 persist_directory: str = None,
                 embedding_model: str = None):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedding_model: OpenAI embedding model to use
        """
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.persist_directory = persist_directory or Config.VECTOR_DB_PATH
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model or Config.EMBEDDING_MODEL
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
    
    def add_documents(self, documents: List[Dict], batch_size: int = 100) -> int:
        """
        Add documents to vector store
        
        Args:
            documents: List of document dictionaries
            batch_size: Number of documents to process at once
            
        Returns:
            Number of chunks added
        """
        texts = []
        metadatas = []
        
        for doc in documents:
            if not doc.get('success', False):
                continue
            
            text = doc.get('text', '')
            if not text or len(text) < Config.MIN_CHUNK_SIZE:
                continue
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    'source': doc.get('url', ''),
                    'title': doc.get('title', ''),
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'authors': str(doc.get('authors', [])),
                    'publish_date': str(doc.get('publish_date', '')),
                    'method': doc.get('method', 'unknown')
                })
        
        # Add to vector store in batches
        total_added = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            if batch_texts:
                self.vector_store.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_texts)
        
        print(f"Added {total_added} chunks to vector store")
        return total_added
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Metadata filter
            
        Returns:
            List of similar documents
        """
        try:
            results = self.vector_store.similarity_search(
                query, 
                k=k,
                filter=filter
            )
            
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': 0  # Chroma doesn't return scores by default
                }
                for doc in results
            ]
            
        except Exception as e:
            print(f"Similarity search error: {e}")
            return []
    
    def similarity_search_with_score(self, 
                                    query: str, 
                                    k: int = 5) -> List[Dict]:
        """
        Search with similarity scores
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of documents with scores
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in results
            ]
            
        except Exception as e:
            print(f"Similarity search with score error: {e}")
            return []
    
    def mmr_search(self, 
                   query: str,
                   k: int = 5,
                   fetch_k: int = 20,
                   lambda_mult: float = 0.5) -> List[Dict]:
        """
        Maximum Marginal Relevance search for diverse results
        
        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of documents to fetch for MMR
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            
        Returns:
            List of diverse documents
        """
        try:
            results = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                for doc in results
            ]
            
        except Exception as e:
            print(f"MMR search error: {e}")
            return []
    
    def search_by_source(self, source: str, k: int = 10) -> List[Dict]:
        """
        Search documents from a specific source
        
        Args:
            source: Source URL or identifier
            k: Number of results
            
        Returns:
            Documents from specified source
        """
        return self.similarity_search(
            query="",
            k=k,
            filter={"source": source}
        )
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary of statistics
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            # Get unique sources
            all_docs = collection.get()
            sources = set()
            if all_docs and 'metadatas' in all_docs:
                for metadata in all_docs['metadatas']:
                    if metadata and 'source' in metadata:
                        sources.add(metadata['source'])
            
            return {
                'total_chunks': count,
                'unique_sources': len(sources),
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
            
        except Exception as e:
            print(f"Stats error: {e}")
            return {}
    
    def delete_by_source(self, source: str) -> bool:
        """
        Delete all documents from a specific source
        
        Args:
            source: Source URL to delete
            
        Returns:
            Success status
        """
        try:
            collection = self.vector_store._collection
            
            # Get IDs with matching source
            results = collection.get(
                where={"source": source}
            )
            
            if results and 'ids' in results and results['ids']:
                collection.delete(ids=results['ids'])
                print(f"Deleted {len(results['ids'])} chunks from {source}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Delete error: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from collection
        
        Returns:
            Success status
        """
        try:
            # Delete the collection
            self.vector_store.delete_collection()
            
            # Recreate it
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            print(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            print(f"Clear collection error: {e}")
            return False
    
    def update_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """
        Update metadata for a document
        
        Args:
            doc_id: Document ID
            metadata: New metadata
            
        Returns:
            Success status
        """
        try:
            collection = self.vector_store._collection
            collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
            return True
            
        except Exception as e:
            print(f"Update metadata error: {e}")
            return False
    
    def export_collection(self, output_path: str) -> bool:
        """
        Export collection data
        
        Args:
            output_path: Path to export to
            
        Returns:
            Success status
        """
        try:
            import json
            
            collection = self.vector_store._collection
            data = collection.get(include=['documents', 'metadatas'])
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Exported collection to {output_path}")
            return True
            
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def search_with_context(self, 
                           query: str,
                           k: int = 5,
                           context_chunks: int = 2) -> List[Dict]:
        """
        Search and include surrounding context chunks
        
        Args:
            query: Search query
            k: Number of results
            context_chunks: Number of chunks before/after to include
            
        Returns:
            Documents with context
        """
        results = self.similarity_search(query, k=k)
        
        enriched_results = []
        for result in results:
            metadata = result['metadata']
            chunk_id = metadata.get('chunk_id', 0)
            source = metadata.get('source', '')
            
            # Get surrounding chunks
            context_before = []
            context_after = []
            
            for i in range(1, context_chunks + 1):
                # Chunk before
                if chunk_id - i >= 0:
                    prev_chunk = self.similarity_search(
                        query="",
                        k=1,
                        filter={
                            'source': source,
                            'chunk_id': chunk_id - i
                        }
                    )
                    if prev_chunk:
                        context_before.insert(0, prev_chunk[0]['content'])
                
                # Chunk after
                next_chunk = self.similarity_search(
                    query="",
                    k=1,
                    filter={
                        'source': source,
                        'chunk_id': chunk_id + i
                    }
                )
                if next_chunk:
                    context_after.append(next_chunk[0]['content'])
            
            enriched_results.append({
                'content': result['content'],
                'metadata': metadata,
                'context_before': context_before,
                'context_after': context_after,
                'full_context': ' '.join(context_before + [result['content']] + context_after)
            })
        
        return enriched_results