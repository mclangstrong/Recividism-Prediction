"""
Configuration classes for Flask application
===========================================
Provides environment-specific configurations for development, production, and testing.
"""
import os
from datetime import timedelta

class Config:
    """Base configuration with common settings"""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'bjmp-recidivism-dev-key-change-in-production'
    
    # Database
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Use absolute path to avoid ambiguity between root and app/ directories
    basedir = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'recidivism.db')
    
    # Session
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    
    # Cache
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    
    # Compression
    COMPRESS_MIMETYPES = ['text/html', 'text/css', 'text/xml', 'application/json', 
                          'application/javascript', 'text/javascript']
    COMPRESS_LEVEL = 6
    COMPRESS_MIN_SIZE = 500
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_STRATEGY = 'fixed-window'
    
    # Logging
    LOG_FILE = 'app.log'
    LOG_MAX_BYTES = 10000000  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # Export Settings
    PDF_PAGE_SIZE = 'letter'
    PDF_AUTHOR = 'BJMP Recidivism Prediction System'
    
    # Pagination
    ITEMS_PER_PAGE = 20
    MAX_ITEMS_PER_PAGE = 100
    
    # Audit Log Retention
    AUDIT_LOG_RETENTION_DAYS = 365  # Keep audit logs for 1 year


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    
    # More verbose caching for development
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 60  # Shorter timeout for testing
    
    # No secure cookies in development
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    
    # Use Redis for caching in production (if available)
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
    CACHE_REDIS_URL = os.environ.get('REDIS_URL')
    
    # Enhanced security for production
    SESSION_COOKIE_SECURE = True  # Requires HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    
    # Use environment variable for SECRET_KEY
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'bjmp-production-change-this-key'


class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DEBUG = True
    
    # Use in-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    # No caching during tests
    CACHE_TYPE = 'null'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration for specified environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
