import sqlite3

DB_PATH = "bank.db"  # database file

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Create users table if not exists
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_number TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    balance INTEGER DEFAULT 0
)
""")

# Insert only 3 users
users = [
    ("8123623741", "Muruga@123", "Muruga S", "muruga.ca@gmail.com", "6513429873", 250000),
    ("8912672463", "Tharunika@123", "Tharunika K", "tharunika3@gmail.com", "9812327638", 420000),
    ("23647126543", "Krishna@123", "Krishna P", "Krishnna4@gmail.com", "9856437865", 300000)
]

c.executemany("""
INSERT OR REPLACE INTO users 
(account_number, password, name, email, phone, balance)
VALUES (?, ?, ?, ?, ?, ?)
""", users)

conn.commit()
conn.close()

print("✅ Only 3 Users Added: Muruga, Tharunika, Krishna")
