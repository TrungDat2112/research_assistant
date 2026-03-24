"""
Configuration module for Research Assistant
Loads environment variables and defines application settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """Main configuration class"""
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CSE_ID: str = os.getenv("GOOGLE_CSE_ID", "")
    
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "chroma_db"))
    REPORTS_DIR: Path = BASE_DIR / "reports"
    CACHE_DIR: Path = BASE_DIR / "cache"
    
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "research_documents")
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    MAX_SCRAPE_PAGES: int = int(os.getenv("MAX_SCRAPE_PAGES", "5"))
    SEARCH_TIMEOUT: int = 10  
    SCRAPE_DELAY: float = 1.0  
    
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
    
    RESEARCHER_MODEL: str = "gpt-4-turbo-preview"
    ANALYZER_MODEL: str = "gpt-4-turbo-preview"
    WRITER_MODEL: str = "gpt-4-turbo-preview"
    
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    REQUEST_TIMEOUT: int = 10
    MAX_RETRIES: int = 3
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MIN_CHUNK_SIZE: int = 100
    
    DEFAULT_REPORT_FORMAT: str = "md"
    INCLUDE_CITATIONS: bool = True
    INCLUDE_STATISTICS: bool = True
    
    REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_ENABLED: bool = True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.REPORTS_DIR,
            cls.CACHE_DIR,
            Path(cls.VECTOR_DB_PATH)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_api_keys(cls) -> dict:
        """Validate that required API keys are set"""
        keys_status = {
            "openai": bool(cls.OPENAI_API_KEY),
            "anthropic": bool(cls.ANTHROPIC_API_KEY),
            "serper": bool(cls.SERPER_API_KEY),
            "google": bool(cls.GOOGLE_API_KEY and cls.GOOGLE_CSE_ID)
        }
        return keys_status
    
    @classmethod
    def get_api_key(cls, service: str) -> Optional[str]:
        """Get API key for a specific service"""
        service_map = {
            "openai": cls.OPENAI_API_KEY,
            "anthropic": cls.ANTHROPIC_API_KEY,
            "serper": cls.SERPER_API_KEY,
            "google": cls.GOOGLE_API_KEY
        }
        return service_map.get(service.lower())
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if minimum required configuration is present"""
        return bool(cls.OPENAI_API_KEY or cls.ANTHROPIC_API_KEY)
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get a summary of current configuration"""
        return {
            "llm_model": cls.LLM_MODEL,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "max_search_results": cls.MAX_SEARCH_RESULTS,
            "max_scrape_pages": cls.MAX_SCRAPE_PAGES,
            "vector_db_path": cls.VECTOR_DB_PATH,
            "debug_mode": cls.DEBUG
        }


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    MAX_SEARCH_RESULTS = 5
    MAX_SCRAPE_PAGES = 3


class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG = False
    LOG_LEVEL = "WARNING"
    RATE_LIMIT_ENABLED = True


class TestConfig(Config):
    """Test-specific configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    VECTOR_DB_PATH = "./test_data/chroma_db"
    MAX_SEARCH_RESULTS = 3
    MAX_SCRAPE_PAGES = 2


def get_config(env: str = "development") -> Config:
    """Get configuration based on environment"""
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "test": TestConfig
    }
    
    config_class = configs.get(env.lower(), DevelopmentConfig)
    config_class.create_directories()
    
    return config_class


config = get_config(os.getenv("ENVIRONMENT", "development"))


Config.create_directories()