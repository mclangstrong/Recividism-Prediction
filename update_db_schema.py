import sqlite3
import os

DB_PATH = os.path.join('app', 'instance', 'recidivism.db')

def update_schema():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Check if columns exist
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [info[1] for info in cursor.fetchall()]

        if 'ml_probability' not in columns:
            print("Adding ml_probability column...")
            cursor.execute("ALTER TABLE predictions ADD COLUMN ml_probability REAL")
        
        if 'rnr_probability' not in columns:
            print("Adding rnr_probability column...")
            cursor.execute("ALTER TABLE predictions ADD COLUMN rnr_probability REAL")
            
        if 'rnr_breakdown' not in columns:
            print("Adding rnr_breakdown column...")
            cursor.execute("ALTER TABLE predictions ADD COLUMN rnr_breakdown TEXT")

        conn.commit()
        print("Schema update completed successfully.")
        
    except Exception as e:
        print(f"Error updating schema: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    update_schema()
