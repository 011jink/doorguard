import sqlite3

DB_PATH = "doorguard.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            duration_sec REAL DEFAULT 0,
            clip_path TEXT,
            snapshot_path TEXT,
            event_type TEXT DEFAULT 'detection',
            face_id INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_image_path TEXT,
            first_seen TEXT,
            last_seen TEXT,
            visit_count INTEGER DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY,
            threshold_sec INTEGER DEFAULT 1,
            notify_enabled INTEGER DEFAULT 1,
            distance_threshold INTEGER DEFAULT 150,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        INSERT OR IGNORE INTO settings (id, threshold_sec, notify_enabled, distance_threshold)
        VALUES (1, 1, 1, 150)
    ''')

    conn.commit()
    conn.close()
    print("DB 초기화 완료")

if __name__ == "__main__":
    init_db()
