import sqlite3

DB_PATH = "bank.db"


# ---------------- DATABASE CONNECTION ----------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- CREATE ALL TABLES ----------------
def create_db():
    conn = get_db()
    c = conn.cursor()

    # USERS TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_number TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            balance INTEGER DEFAULT 0
        )
    """)

    # CHAT LOGS TABLE (includes intent & confidence)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account TEXT,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            intent TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # TRANSACTIONS TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_account TEXT,
            receiver_account TEXT,
            receiver_name TEXT,
            amount INTEGER,
            mode TEXT,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ADMIN TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL
        )
    """)

    # FAQ TABLE
    c.execute("""
        CREATE TABLE IF NOT EXISTS faq (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Database Ready (users, admin, faq, chat_logs, transactions).")


# ---------------- USER LOGIN ----------------
def verify_user_login(email, password):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
    row = c.fetchone()
    conn.close()
    return row


# ---------------- ADMIN LOGIN ----------------
def verify_admin_login(email, password):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM admin WHERE email=? AND password=?", (email, password))
    row = c.fetchone()
    conn.close()
    return row


# ---------------- GET USER DETAILS ----------------
def get_user_by_account(account):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE account_number=?", (account,))
    row = c.fetchone()
    conn.close()
    return row


# ---------------- BALANCE FUNCTIONS ----------------
def get_balance(account):
    user = get_user_by_account(account)
    if user:
        return user["balance"]
    return None


def update_balance(account, new_balance):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE users SET balance=? WHERE account_number=?", (new_balance, account))
    conn.commit()
    conn.close()


# ---------------- TRANSFER FUNDS ----------------
def transfer_funds(sender_account, receiver_account, amount):
    sender_balance = get_balance(sender_account)
    if sender_balance is None or sender_balance < amount:
        return False, "Insufficient Balance"

    receiver = get_user_by_account(receiver_account)
    if not receiver:
        return False, "Receiver account does not exist"

    update_balance(sender_account, sender_balance - amount)
    update_balance(receiver_account, receiver["balance"] + amount)
    return True, "Transfer Successful"


# ---------------- RECORD TRANSACTION ----------------
def record_transaction(sender, receiver, receiver_name, amount, mode, status):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO transactions (sender_account, receiver_account, receiver_name, amount, mode, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (sender, receiver, receiver_name, amount, mode, status))
    conn.commit()
    conn.close()


# ---------------- SAVE CHAT (with intent & confidence) ----------------
def save_chat(account, user_message, bot_response, intent=None, confidence=None):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO chat_logs (account, user_message, bot_response, intent, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (account, user_message, bot_response, intent, confidence))
    conn.commit()
    conn.close()


# ---------------- FETCH USER TRANSACTIONS ----------------
def get_transactions(account):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT timestamp, sender_account, receiver_account, receiver_name, amount, mode, status
        FROM transactions
        WHERE sender_account=? OR receiver_account=?
        ORDER BY id DESC
    """, (account, account))
    rows = c.fetchall()
    conn.close()

    formatted = []
    for t in rows:
        if t["sender_account"] == account:
            txn_type = f"Sent to {t['receiver_name']}"
        else:
            sender = get_user_by_account(t["sender_account"])
            sender_name = sender["name"] if sender else "Unknown"
            txn_type = f"Received from {sender_name}"

        formatted.append({
            "date": t["timestamp"],
            "type": txn_type,
            "amount": t["amount"],
            "mode": t["mode"],
            "status": t["status"]
        })
    return formatted


# ---------------- DASHBOARD ANALYTICS ----------------
def get_total_queries():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM chat_logs")
    total = c.fetchone()[0]
    conn.close()
    return total


def get_total_intents():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT intent) FROM chat_logs WHERE intent IS NOT NULL AND intent != ''")
    total = c.fetchone()[0]
    conn.close()
    return total


def get_recent_chats(limit=5):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT account, user_message, bot_response, intent, confidence, timestamp
        FROM chat_logs
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return rows


# ---------------- FAQ MANAGEMENT ----------------
def get_all_faqs():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM faq ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows


def add_faq(question, answer):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO faq (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()


def delete_faq(faq_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM faq WHERE id=?", (faq_id,))
    conn.commit()
    conn.close()


# ---------------- SAFE MIGRATION (add missing columns) ----------------
def ensure_columns():
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE chat_logs ADD COLUMN intent TEXT;")
        c.execute("ALTER TABLE chat_logs ADD COLUMN confidence REAL;")
        print("✅ Columns intent & confidence added successfully.")
    except sqlite3.OperationalError:
        print("ℹ️ Columns already exist, skipping ALTER TABLE.")
    conn.commit()
    conn.close()


# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    create_db()
    ensure_columns()
    print("✅ Database setup complete (with intent & confidence).")
