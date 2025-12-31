"""
Database migration script to add new Officer columns.
Run this script once to update the existing database schema.
"""
import sqlite3
import os

def migrate_database():
    """Add new columns to officers table if they don't exist."""
    # Path to the database
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'recidivism.db')
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return False
    
    print(f"Migrating database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check existing columns in officers table
        cursor.execute("PRAGMA table_info(officers)")
        existing_columns = [row[1] for row in cursor.fetchall()]
        print(f"Existing columns: {existing_columns}")
        
        # Add password_changed_at column if it doesn't exist
        if 'password_changed_at' not in existing_columns:
            print("Adding password_changed_at column...")
            cursor.execute("ALTER TABLE officers ADD COLUMN password_changed_at DATETIME")
            print("✓ Added password_changed_at column")
        else:
            print("✓ password_changed_at column already exists")
        
        # Add must_change_password column if it doesn't exist
        if 'must_change_password' not in existing_columns:
            print("Adding must_change_password column...")
            cursor.execute("ALTER TABLE officers ADD COLUMN must_change_password BOOLEAN DEFAULT 0")
            print("✓ Added must_change_password column")
        else:
            print("✓ must_change_password column already exists")
        
        conn.commit()
        print("\n✅ Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()

if __name__ == '__main__':
    migrate_database()
