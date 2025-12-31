"""
Database backup utilities for the Recidivism Prediction System.
Provides functionality to create and download database backups.
"""
import os
import shutil
from datetime import datetime
from flask import current_app
import logging

logger = logging.getLogger('recidivism_app')


def get_database_path():
    """Get the path to the SQLite database file."""
    # Parse the database URI from config
    db_uri = current_app.config.get('SQLALCHEMY_DATABASE_URI', '')
    
    if db_uri.startswith('sqlite:///'):
        db_path = db_uri.replace('sqlite:///', '')
        return db_path
    
    return None


def create_backup(backup_dir: str = None) -> tuple:
    """
    Create a backup of the SQLite database.
    
    Args:
        backup_dir: Directory to store backup. Defaults to 'backups' folder.
        
    Returns:
        Tuple of (success: bool, backup_path: str or error_message: str)
    """
    try:
        db_path = get_database_path()
        
        if not db_path or not os.path.exists(db_path):
            return (False, "Database file not found")
        
        # Create backup directory if not specified
        if backup_dir is None:
            backup_dir = os.path.join(os.path.dirname(db_path), '..', 'backups')
        
        os.makedirs(backup_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'recidivism_backup_{timestamp}.db'
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy database file
        shutil.copy2(db_path, backup_path)
        
        logger.info(f"Database backup created: {backup_path}")
        
        return (True, backup_path)
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return (False, str(e))


def get_backup_info(backup_path: str) -> dict:
    """Get information about a backup file."""
    if not os.path.exists(backup_path):
        return None
    
    stat = os.stat(backup_path)
    return {
        'filename': os.path.basename(backup_path),
        'path': backup_path,
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat()
    }


def list_backups(backup_dir: str = None) -> list:
    """
    List all available backups.
    
    Args:
        backup_dir: Directory containing backups.
        
    Returns:
        List of backup info dictionaries.
    """
    try:
        if backup_dir is None:
            db_path = get_database_path()
            if db_path:
                backup_dir = os.path.join(os.path.dirname(db_path), '..', 'backups')
            else:
                return []
        
        if not os.path.exists(backup_dir):
            return []
        
        backups = []
        for filename in os.listdir(backup_dir):
            if filename.endswith('.db') and filename.startswith('recidivism_backup_'):
                backup_path = os.path.join(backup_dir, filename)
                info = get_backup_info(backup_path)
                if info:
                    backups.append(info)
        
        # Sort by creation date (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        
        return backups
        
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        return []


def cleanup_old_backups(backup_dir: str = None, keep_count: int = 10) -> int:
    """
    Remove old backups, keeping only the most recent ones.
    
    Args:
        backup_dir: Directory containing backups.
        keep_count: Number of backups to keep.
        
    Returns:
        Number of backups deleted.
    """
    try:
        backups = list_backups(backup_dir)
        
        if len(backups) <= keep_count:
            return 0
        
        # Delete older backups
        deleted_count = 0
        for backup in backups[keep_count:]:
            try:
                os.remove(backup['path'])
                deleted_count += 1
                logger.info(f"Deleted old backup: {backup['filename']}")
            except Exception as e:
                logger.error(f"Error deleting backup {backup['filename']}: {e}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error cleaning up backups: {e}")
        return 0
