from flask import (
    Flask, render_template, request, jsonify, send_file,
    redirect, url_for, session, flash
)

import sqlite3
import csv
from datetime import datetime, timedelta
import os
import traceback
import importlib  # for retraining milestone_two
import pandas as pd

# BOT LOGIC (Milestone 2)
import milestone_two as bot

# DB FUNCTIONS
from db import (
    get_db,
    get_user_by_account,
    verify_user_login,
    verify_admin_login,
    get_balance,
    update_balance,
    transfer_funds,
    save_chat,
    get_transactions,
    get_total_queries,
    get_total_intents,
    get_recent_chats,
    get_all_faqs,
    add_faq,
    delete_faq
)

# ---------------- FLASK CONFIG ----------------
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bank.db")

# Training CSV used by milestone_two
TRAINING_FILE = os.path.join(BASE_DIR, "bankbot_final_expanded1.csv")
APP_TRAINED_AT = datetime.now()

# ---------------- RESET BOT CONTEXT ----------------
def reset_all_bot_context():
    for fn in ("reset_card", "reset_atm", "reset_loan", "reset_acct"):
        f = getattr(bot, fn, None)
        if callable(f):
            try:
                f()
            except:
                pass

    try:
        bot.memory["flow"] = None
        bot.memory["step"] = 0
    except:
        pass


# ---------------- LOGIN CHECK ----------------
def require_login():
    if not session.get("account"):
        return False
    try:
        bot.memory["current_user_account"] = session["account"]
    except:
        pass
    return True


# ---------------- HOME ----------------
@app.route("/")
def admin_home():
    return render_template("admin_home.html")


# ---------------- USER LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        try:
            user = verify_user_login(email, password)
        except Exception:
            traceback.print_exc()
            user = None

        if user:
            # sqlite3.Row uses dict-style access, not .get()
            session["account"] = user["account_number"]
            session["email"] = user["email"]
            session["name"] = user["name"]
            session["balance"] = user["balance"]

            try:
                bot.memory["current_user_account"] = user["account_number"]
            except:
                pass

            reset_all_bot_context()
            return redirect(url_for("dashboard"))

        flash("❌ Invalid Email or Password.", "error")

    return render_template("Login.html")


# ---------------- USER DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if not require_login():
        return redirect(url_for("login"))

    user = get_user_by_account(session["account"])
    transactions = get_transactions(session["account"])

    return render_template(
        "dashboard.html",
        name=user["name"],
        account=user["account_number"],
        balance=user["balance"],
        email=user["email"],
        phone=user["phone"],
        transactions=transactions
    )


# ---------------- CHAT WINDOW ----------------
@app.route("/chat")
def chat():
    if not require_login():
        return redirect(url_for("login"))
    return render_template("chat.html", account=session["account"])


# ---------------- RESET CONTEXT ----------------
@app.route("/reset_context", methods=["POST"])
def reset_context():
    if not require_login():
        return ("", 401)
    reset_all_bot_context()
    return ("", 204)


# ---------------- GET BOT RESPONSE (Milestone 2 + ML override) ----------------
@app.route("/get_response", methods=["POST"])
def get_response():
    if not require_login():
        return jsonify({"response": "Please login first."}), 401

    msg = request.json.get("message", "").strip()
    if not msg:
        return jsonify({"response": "Please type something."})

    bot.memory["current_user_account"] = session["account"]

    # ---- Step 1: Milestone-2 logic ----
    try:
        result = bot.handle_user_input(msg)

        if isinstance(result, (tuple, list)):
            if len(result) == 3:
                intent, entities, reply = result
                confidence = 0.70
            elif len(result) == 4:
                intent, entities, reply, confidence = result
            else:
                intent, entities, reply, confidence = "unknown", {}, "Unexpected bot output.", 0.50
        else:
            intent, entities, reply, confidence = "unknown", {}, str(result), 0.50

    except Exception as e:
        print("BOT ERROR:", e)
        intent, entities, reply, confidence = "error", {}, "Server error.", 0.0

    # ---- Step 2: ML model override ----
    try:
        if hasattr(bot, "model") and bot.model is not None:
            baseline_intent = intent  # from rule-based flow

            # Only use ML when rule-based is confused
            if baseline_intent in ("unknown", "general_banking_info", "out_of_scope"):
                ml_pred = bot.model.predict([msg])[0]
                ml_prob = float(max(bot.model.predict_proba([msg])[0]))

                intent = ml_pred
                confidence = ml_prob

    except Exception as e:
        print("ML ERROR:", e)


    # ---- Step 3: Save chat ----
    try:
        save_chat(session["account"], msg, reply, intent, float(confidence))
    except Exception as e:
        print("DB SAVE ERROR:", e)

    return jsonify({
        "response": reply,
        "intent": intent,
        "confidence": confidence,
        "entities": entities
    })


# ---------------- CHAT LOGS (USER VIEW) ----------------
@app.route("/chat_logs")
def chat_logs():
    if not require_login():
        return redirect(url_for("login"))

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT user_message, bot_response, timestamp
        FROM chat_logs WHERE account=?
        ORDER BY id DESC
    """, (session["account"],))
    rows = c.fetchall()
    conn.close()

    formatted = []
    for r in rows:
        t = r["timestamp"]
        try:
            t = datetime.strptime(t, "%Y-%m-%d %H:%M:%S") + timedelta(hours=5, minutes=30)
            t = t.strftime("%d %b %Y - %I:%M %p")
        except:
            pass
        formatted.append((r["user_message"], r["bot_response"], t))

    return render_template("chat_logs.html", logs=formatted, account=session["account"])


# ---------------- EXPORT CHAT LOGS (CSV) ----------------
@app.route("/export_excel")
def export_excel():
    conn = get_db()
    c = conn.cursor()

    if session.get("admin"):
        filename = "all_chat_logs.csv"
        filepath = os.path.join(BASE_DIR, filename)
        c.execute("""
            SELECT account, user_message, bot_response, intent, confidence, timestamp
            FROM chat_logs ORDER BY id
        """)
    else:
        filename = f"{session['account']}_chat_log.csv"
        filepath = os.path.join(BASE_DIR, filename)
        c.execute("""
            SELECT user_message, bot_response, intent, confidence, timestamp
            FROM chat_logs WHERE account=?
            ORDER BY id
        """, (session["account"],))

    data = c.fetchall()
    conn.close()

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if session.get("admin"):
            writer.writerow(["Account", "User Message", "Bot Response", "Intent", "Confidence", "Time"])
            for r in data:
                writer.writerow([
                    r["account"],
                    r["user_message"],
                    r["bot_response"],
                    r["intent"],
                    r["confidence"],
                    r["timestamp"]
                ])
        else:
            writer.writerow(["User Message", "Bot Response", "Intent", "Confidence", "Time"])
            for r in data:
                writer.writerow([
                    r["user_message"],
                    r["bot_response"],
                    r["intent"],
                    r["confidence"],
                    r["timestamp"]
                ])

    return send_file(filepath, as_attachment=True)


# ---------------- ADMIN LOGIN ----------------
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        admin = verify_admin_login(username, password)

        if admin:
            session["admin"] = True
            session["admin_name"] = admin["name"]
            session["admin_email"] = admin["email"]
            return redirect(url_for("admin_dashboard"))

        # fallback login (from setup_admin.py)
        if username == "admin@caashmora.ac.in" and password == "admin@123":
            session["admin"] = True
            session["admin_name"] = "System Administrator"
            session["admin_email"] = username
            return redirect(url_for("admin_dashboard"))

        flash("❌ Invalid Admin Email or Password", "error")

    return render_template("admin_login.html")


# ---------------- ADMIN DASHBOARD ----------------
@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    # ---------- DATASET STATS & TRAINING ACCURACY ----------
    try:
        # use the same df & model that milestone_two trained
        df = bot.df           # pandas DataFrame from milestone_two
        model = bot.model     # trained pipeline

        # total queries & intents from dataset
        total_queries = len(df)
        total_intents = df["intent"].astype(str).nunique()

        # training accuracy: how many predictions match the true intents
        X = df["text"].astype(str)
        y = df["intent"].astype(str)
        preds = model.predict(X)
        acc_value = (preds == y).mean() * 100.0
        accuracy = f"{acc_value:.1f}%"

    except Exception as e:
        print("DASHBOARD DATASET / ACCURACY ERROR:", e)
        # fallback to DB-based counts if something goes wrong
        total_queries = get_total_queries()
        total_intents = get_total_intents()
        accuracy = "N/A"

    # ---------- LAST RETRAINED TIME ----------
    last_retrained = APP_TRAINED_AT.strftime("%d-%m-%Y %I:%M:%S %p")

    # ---------- RECENT USER QUERIES TABLE ----------
    recent = get_recent_chats(limit=5)
    formatted = []
    for r in recent:
        conf = r["confidence"]

        if conf is None:
            conf_str = "—%"
        else:
            # confidence is stored as 0–1 in DB; show as percentage
            conf_str = f"{float(conf) * 100:.1f}%"

        formatted.append({
            "account": r["account"],
            "user_message": r["user_message"],
            "intent": r["intent"],
            "confidence": conf_str,
            "timestamp": r["timestamp"]
        })

    return render_template(
        "admin_dashboard.html",
        total_queries=total_queries,
        total_intents=total_intents,
        accuracy=accuracy,
        last_retrained=last_retrained,
        recent_queries=formatted
    )


# ---------------- ADMIN: QUERIES (simple view) ----------------
@app.route("/admin_queries")
def admin_queries():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT account, user_message, bot_response, timestamp FROM chat_logs ORDER BY id DESC")
    queries = c.fetchall()
    conn.close()
    
    return render_template("admin_queries.html", queries=queries)


# ---------------- ADMIN: FAQ MANAGEMENT (ADD + DELETE) ----------------
@app.route("/admin_faq", methods=["GET", "POST"])
def admin_faq():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        answer = request.form.get("answer", "").strip()

        if question and answer:
            add_faq(question, answer)
            flash("✅ FAQ added successfully.", "success")
        else:
            flash("⚠️ Please enter both question and answer.", "error")

        return redirect(url_for("admin_faq"))

    faqs = get_all_faqs()
    return render_template("admin_faq.html", faqs=faqs)

@app.route("/add_faq", methods=["POST"])
def add_faq_route():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    question = request.form.get("question", "").strip()
    answer = request.form.get("answer", "").strip()

    if question and answer:
        add_faq(question, answer)
        flash("✔ FAQ added successfully!", "success")
    else:
        flash("⚠ Both fields required!", "error")

    return redirect(url_for("admin_faq"))


@app.post("/admin_faq/delete/<int:faq_id>")
def admin_faq_delete(faq_id):
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    delete_faq(faq_id)
    flash("🗑️ FAQ deleted.", "success")
    return redirect(url_for("admin_faq"))


# ---------------- ADMIN: TRAINING DATA HELPER ----------------
def append_training_sample(text, intent, response):
    """
    Append a new training row to the CSV used by milestone_two.
    """
    # Ensure file exists with header. If missing, create with basic header.
    file_exists = os.path.exists(TRAINING_FILE)
    with open(TRAINING_FILE, "a", newline="", encoding="latin1") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "intent", "response"])
        writer.writerow([text, intent, response])


# ---------------- ADMIN: TRAINING PAGE (view + add samples) ----------------
@app.route("/admin_training", methods=["GET", "POST"])
def admin_training():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        # Form fields from admin_training.html
        text = request.form.get("text", "").strip()
        intent = request.form.get("intent", "").strip()
        response = request.form.get("response", "").strip()

        if text and intent and response:
            append_training_sample(text, intent, response)
            flash("✅ Training example added to dataset.", "success")
        else:
            flash("⚠️ Please fill all fields (text, intent, response).", "error")

        return redirect(url_for("admin_training"))

    faqs = get_all_faqs()
    recent_chats = get_recent_chats(limit=10)

    # You can show recent chats so admin can copy text/intent/response
    return render_template("admin_training.html", faqs=faqs, recent_chats=recent_chats)


# ---------------- ADMIN: FULL CHAT LOGS VIEW ----------------
# ---------------- ADMIN: FULL CHAT LOGS VIEW ----------------
@app.route("/admin_chatlogs")
def admin_chatlogs():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    
    conn = get_db()
    c = conn.cursor()
    # admin sees ALL chats, select only 3 fields (no id)
    c.execute("""
        SELECT user_message, bot_response, timestamp
        FROM chat_logs
        ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()

    # format exactly like user /chat_logs route
    formatted = []
    for r in rows:
        t = r["timestamp"]
        try:
            t = datetime.strptime(t, "%Y-%m-%d %H:%M:%S") + timedelta(hours=5, minutes=30)
            t = t.strftime("%d %b %Y - %I:%M %p")
        except:
            pass
        formatted.append((r["user_message"], r["bot_response"], t))

    # reuse the SAME template
    return render_template("chat_logs.html", logs=formatted, account="ADMIN")



# ---------------- ADMIN: RETRAIN MODEL FROM UI ----------------
@app.post("/admin_retrain")
def admin_retrain():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    try:
        global bot
        bot = importlib.reload(bot)   # your existing retrain logic
        reset_all_bot_context()

        # save real retrain time
        with open("last_retrained.txt", "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        flash("✅ Model retrained from latest training data.", "success")
    except Exception as e:
        print("RETRAIN ERROR:", e)
        flash("❌ Failed to retrain model. Check server logs.", "error")

    return redirect(url_for("admin_dashboard"))

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    reset_all_bot_context()
    session.clear()
    return redirect(url_for("admin_home"))


# ---------------- 404 HANDLER ----------------
@app.errorhandler(404)
def not_found(_):
    return redirect(url_for("admin_home"))


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    try:
        from db import create_db
        create_db()
    except:
        pass

    app.run(debug=True, port=5000)
