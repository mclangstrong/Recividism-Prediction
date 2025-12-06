"""
Audit logging decorators and utilities for the Recidivism Prediction System.
Tracks all critical operations for compliance and accountability.
"""
from functools import wraps
from flask import session, request
from datetime import datetime
import json


def log_audit(action_type, description=None, model_version=None, metadata=None):
    """
    Decorator to automatically log operations to the audit trail.
    
    Usage:
        @log_audit('prediction', 'Created new risk assessment')
        def some_function():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Import here to avoid circular dependency
            from app.database import db, AuditLog
            
            # Execute the function first
            result = f(*args, **kwargs)
            
            try:
                # Create audit log entry
                audit = AuditLog(
                    action_type=action_type,
                    action_description=description or f.__name__,
                    username=session.get('username', 'Unknown'),
                    user_id=session.get('user_id'),
                    model_version=model_version or '1.0',
                    ip_address=request.remote_addr if request else None,
                    extra_data=json.dumps(metadata) if metadata else None
                )
                db.session.add(audit)
                db.session.commit()
            except Exception as e:
                # Don't let audit logging failure crash the app
                print(f"Audit logging error: {e}")
            
            return result
        return decorated_function
    return decorator


def create_audit_log(action_type, description, prediction_id=None, pdl_id=None, 
                     model_version=None, metadata=None):
    """
    Manually create an audit log entry.
    Use this when you need more control than the decorator provides.
    """
    from app.database import db, AuditLog
    from flask import session, request
    
    try:
        audit = AuditLog(
            action_type=action_type,
            action_description=description,
            username=session.get('username', 'System'),
            user_id=session.get('user_id'),
            prediction_id=prediction_id,
            pdl_id=pdl_id,
            model_version=model_version or '1.0',
            ip_address=request.remote_addr if request else None,
            extra_data=json.dumps(metadata) if metadata else None
        )
        db.session.add(audit)
        db.session.commit()
        return audit.id
    except Exception as e:
        print(f"Audit logging error: {e}")
        return None
