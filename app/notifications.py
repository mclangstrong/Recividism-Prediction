"""
In-app notification system for the Recidivism Prediction System.
NO email functionality - all notifications are in-app only for security.
"""
from datetime import datetime
from flask import session


def create_notification(notification_type, title, message, prediction_id=None, pdl_id=None, user_id=None):
    """
    Create an in-app notification (NO email).
    
    Args:
        notification_type: 'high_risk', 'system', 'warning', etc.
        title: Short notification title
        message: Detailed notification message
        prediction_id: Optional prediction ID
        pdl_id: Optional PDL ID
        user_id: Target user (None = all users)
    
    Returns:
        Notification ID if successful, None otherwise
    """
    from database import db, Notification
    
    try:
        notification = Notification(
            notification_type=notification_type,
            title=title,
            message=message,
            prediction_id=prediction_id,
            pdl_id=pdl_id,
            user_id=user_id,
            is_read=False
        )
        db.session.add(notification)
        db.session.commit()
        return notification.id
    except Exception as e:
        print(f"Notification creation error: {e}")
        return None


def create_high_risk_notification(pdl_name, pdl_id, probability, prediction_id):
    """
    Create a high-risk alert notification when a high-risk assessment is made.
    """
    title = f"🚨 High Risk Alert: {pdl_name}"
    message = f"A high-risk assessment has been completed for {pdl_name} (ID: {pdl_id}). " \
              f"Recidivism probability: {int(probability * 100)}%. Immediate attention recommended."
    
    return create_notification(
        notification_type='high_risk',
        title=title,
        message=message,
        prediction_id=prediction_id,
        pdl_id=pdl_id,
        user_id=None  # Broadcast to all users
    )


def mark_notification_as_read(notification_id, user_id=None):
    """Mark a notification as read."""
    from database import db, Notification
    
    try:
        notification = Notification.query.get(notification_id)
        if notification:
            notification.is_read = True
            notification.read_at = datetime.utcnow()
            db.session.commit()
            return True
    except Exception as e:
        print(f"Error marking notification as read: {e}")
    return False


def get_unread_count(user_id=None):
    """Get count of unread notifications for a user (or all if user_id is None)."""
    from database import Notification
    
    try:
        query = Notification.query.filter_by(is_read=False)
        if user_id:
            query = query.filter((Notification.user_id == user_id) | (Notification.user_id == None))
        return query.count()
    except Exception as e:
        print(f"Error getting unread count: {e}")
        return 0


def get_recent_notifications(user_id=None, limit=10):
    """Get recent notifications for a user."""
    from database import Notification
    
    try:
        query = Notification.query
        if user_id:
            query = query.filter((Notification.user_id == user_id) | (Notification.user_id == None))
        
        notifications = query.order_by(Notification.timestamp.desc()).limit(limit).all()
        return [n.to_dict() for n in notifications]
    except Exception as e:
        print(f"Error getting notifications: {e}")
        return []
