"""
Custom error handlers for Flask application
===========================================
Provides user-friendly error pages and proper error logging.
"""
from flask import render_template, jsonify, request
import logging

logger = logging.getLogger(__name__)


def register_error_handlers(app):
    """Register all error handlers with the Flask app"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 - Page Not Found errors"""
        logger.warning(f"404 error: {request.url}")
        
        if request.path.startswith('/api/'):
            # Return JSON for API requests
            return jsonify({
                'error': 'Resource not found',
                'status': 404
            }), 404
        
        # Return HTML page for web requests
        return render_template('errors/404.html'), 404
    
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 - Internal Server Error"""
        logger.error(f"500 error: {error}", exc_info=True)
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Internal server error',
                'status': 500
            }), 500
        
        return render_template('errors/500.html'), 500
    
    
    @app.errorhandler(403)
    def forbidden_error(error):
        """Handle 403 - Forbidden errors"""
        logger.warning(f"403 error: {request.url}")
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Forbidden',
                'status': 403
            }), 403
        
        return render_template('errors/403.html'), 403
    
    
    @app.errorhandler(401)
    def unauthorized_error(error):
        """Handle 401 - Unauthorized errors"""
        logger.warning(f"401 error: {request.url}")
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Unauthorized',
                'status': 401
            }), 401
        
        return render_template('errors/401.html'), 401
    
    
    @app.errorhandler(429)
    def ratelimit_handler(error):
        """Handle 429 - Too Many Requests (rate limiting)"""
        logger.warning(f"Rate limit exceeded: {request.url}")
        
        if request.path.startswith('/api/'):
            return jsonify({
                'error': 'Rate limit exceeded. Please try again later.',
                'status': 429
            }), 429
        
        return render_template('errors/429.html'), 429
    
    
    return app
