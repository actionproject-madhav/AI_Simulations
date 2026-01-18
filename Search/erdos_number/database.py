import sqlite3
import os
from typing import Set, Optional

DB_PATH = "wiki_cache.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Table for outgoing links (page -> link)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS outgoing_links (
            source TEXT,
            target TEXT,
            PRIMARY KEY (source, target)
        )
    """)
    # Table for incoming links (target -> source)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incoming_links (
            target TEXT,
            source TEXT,
            PRIMARY KEY (target, source)
        )
    """)
    # Table to track if a page has been fully cached
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cached_pages (
            title TEXT PRIMARY KEY,
            type TEXT -- 'outgoing' or 'incoming'
        )
    """)
    conn.commit()
    conn.close()

def get_cached_links(title: str, incoming: bool = False) -> Optional[Set[str]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    ctype = 'incoming' if incoming else 'outgoing'
    cursor.execute("SELECT 1 FROM cached_pages WHERE title = ? AND type = ?", (title, ctype))
    if not cursor.fetchone():
        conn.close()
        return None
        
    if incoming:
        cursor.execute("SELECT source FROM incoming_links WHERE target = ?", (title,))
    else:
        cursor.execute("SELECT target FROM outgoing_links WHERE source = ?", (title,))
        
    links = {row[0] for row in cursor.fetchall()}
    conn.close()
    return links

def save_links(title: str, links: Set[str], incoming: bool = False):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    ctype = 'incoming' if incoming else 'outgoing'
    table = 'incoming_links' if incoming else 'outgoing_links'
    col1, col2 = ('target', 'source') if incoming else ('source', 'target')
    
    # Batch insert
    data = [(title, link) for link in links]
    cursor.executemany(f"INSERT OR IGNORE INTO {table} ({col1}, {col2}) VALUES (?, ?)", data)
    cursor.execute("INSERT OR REPLACE INTO cached_pages (title, type) VALUES (?, ?)", (title, ctype))
    
    conn.commit()
    conn.close()

# Initialize on import
if not os.path.exists(DB_PATH):
    init_db()
