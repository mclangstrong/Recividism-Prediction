"""
Flask extensions initialization
===============================
Centralizes the initialization of Flask extensions for better organization.
"""
from flask_caching import Cache
from flask_compress import Compress

# Initialize extensions
cache = Cache()
compress = Compress()


def init_extensions(app):
    """Initialize all Flask extensions with the app"""
    cache.init_app(app)
    compress.init_app(app)
    
    return app
