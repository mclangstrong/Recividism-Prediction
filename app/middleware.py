"""
Performance monitoring middleware
==================================
Tracks request/response times and logs slow queries.
"""
import time
import logging
from flask import request, g

logger = logging.getLogger(__name__)


def setup_request_timing(app):
    """Add request timing middleware to track performance"""
    
    @app.before_request
    def before_request():
        """Start timing the request"""
        g.start_time = time.time()
    
    
    @app.after_request
    def after_request(response):
        """Log request duration and add timing header"""
        if hasattr(g, 'start_time'):
            elapsed = time.time() - g.start_time
            
            # Add timing header
            response.headers['X-Response-Time'] = f"{elapsed:.3f}s"
            
            # Log slow requests (>1 second)
            if elapsed > 1.0:
                logger.warning(
                    f"Slow request: {request.method} {request.path} "
                    f"took {elapsed:.3f}s"
                )
            
            # Log all requests in debug mode
            elif app.debug:
                logger.debug(
                    f"{request.method} {request.path} - {response.status_code} "
                    f"in {elapsed:.3f}s"
                )
        
        return response
    
    
    return app


def add_security_headers(app):
    """Add security headers to all responses"""
    
    @app.after_request
    def set_security_headers(response):
        """Add security headers"""
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'SAN EORIGIN'
        
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Enable XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "img-src * data:;"
        )
        
        # Referrer policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        return response
    
    return app
