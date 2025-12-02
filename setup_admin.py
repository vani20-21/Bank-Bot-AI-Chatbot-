import sqlite3

DB_PATH = "bank.db"

def setup_admin():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create admin table if not exists
    c.execute("""
    CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL
    )
    """)

    # Insert your professional admin credentials
    admin_email = "admin@caashmora.ac.in"
    admin_password = "admin@123"
    admin_name = "System Administrator"

    c.execute("""
    INSERT OR REPLACE INTO admin (id, email, password, name)
    VALUES (1, ?, ?, ?)
    """, (admin_email, admin_password, admin_name))

    conn.commit()
    conn.close()
    print("✅ Admin account added successfully!")
    print(f"Email: {admin_email}")
    print(f"Password: {admin_password}")

if __name__ == "__main__":
    setup_admin()
