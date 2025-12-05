from flask import (
    Flask, render_template, request, jsonify, send_file,
    redirect, url_for, session, flash
)

import csv
from datetime import datetime, timedelta
import os
import traceback
import importlib
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
    delete_faq,
    get_frequent_questions,      # auto FAQ from chat_logs
    create_db,
    ensure_columns,
)

# ---------------- FLASK CONFIG ----------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_FILE = os.path.join(BASE_DIR, "bankbot_final_expanded1.csv")
APP_TRAINED_AT = datetime.now()


# ---------------- RESET BOT CONTEXT ----------------
def reset_all_bot_context():
    try:
        bot.memory.clear()
    except Exception:
        pass


# ---------------- LOGIN CHECK (USER) ----------------
def require_login():
    if not session.get("account"):
        return False
    try:
        bot.memory["current_user_account"] = session["account"]
    except Exception:
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
            session["account"] = user["account_number"]
            session["email"] = user["email"]
            session["name"] = user["name"]
            session["balance"] = user["balance"]

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
    txns = get_transactions(session["account"])

    return render_template(
        "dashboard.html",
        name=user["name"],
        account=user["account_number"],
        balance=user["balance"],
        email=user["email"],
        phone=user["phone"],
        transactions=txns,
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


# ---------------- GET BOT RESPONSE ----------------
@app.route("/get_response", methods=["POST"])
def get_response():
    if not require_login():
        return jsonify({"response": "Please login first."}), 401

    msg = request.json.get("message", "").strip()
    if not msg:
        return jsonify({"response": "Please type something."})

    # ---- Step 1: Rule-based (Milestone 2) ----
    try:
        result = bot.handle_user_input(msg)

        if isinstance(result, (tuple, list)):
            if len(result) == 3:
                intent, entities, reply = result
                confidence = 0.70
            elif len(result) == 4:
                intent, entities, reply, confidence = result
            else:
                intent, entities, reply, confidence = (
                    "unknown",
                    {},
                    "Unexpected bot output.",
                    0.50,
                )
        else:
            intent, entities, reply, confidence = "unknown", {}, str(result), 0.50
    except Exception as e:
        print("BOT ERROR:", e)
        intent, entities, reply, confidence = "error", {}, "Server error.", 0.0

    # ---- Step 2: ML override when rule-based is confused ----
    try:
        if hasattr(bot, "model") and bot.model is not None:
            baseline_intent = intent

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

    return jsonify(
        {
            "response": reply,
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
        }
    )


# ---------------- USER CHAT LOGS (only if needed) ----------------
@app.route("/chat_logs")
def chat_logs():
    if not require_login():
        return redirect(url_for("login"))

    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        SELECT user_message, bot_response, timestamp
        FROM chat_logs
        WHERE account=?
        ORDER BY id DESC
    """,
        (session["account"],),
    )
    rows = c.fetchall()
    conn.close()

    formatted = []
    for r in rows:
        t = r["timestamp"]
        try:
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            t = dt + timedelta(hours=5, minutes=30)
            t = t.strftime("%d %b %Y - %I:%M %p")
        except Exception:
            pass
        formatted.append((r["user_message"], r["bot_response"], t))

    return render_template("chat_logs.html", logs=formatted, account=session["account"])


# ---------------- EXPORT MESSAGE LOGS ----------------
@app.route("/export_excel")
def export_excel():
    conn = get_db()
    c = conn.cursor()

    if session.get("admin"):
        filename = "all_chat_logs.csv"
        filepath = os.path.join(BASE_DIR, filename)
        c.execute(
            """
            SELECT account, user_message, bot_response, intent, confidence, timestamp
            FROM chat_logs
            ORDER BY id
        """
        )
    else:
        filename = f"{session['account']}_chat_log.csv"
        filepath = os.path.join(BASE_DIR, filename)
        c.execute(
            """
            SELECT user_message, bot_response, intent, confidence, timestamp
            FROM chat_logs
            WHERE account=?
            ORDER BY id
        """,
            (session["account"],),
        )

    rows = c.fetchall()
    conn.close()

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if rows:
            writer.writerow(rows[0].keys())
        for r in rows:
            writer.writerow(list(r))

    return send_file(filepath, as_attachment=True)


# ---------------- ADMIN LOGIN ----------------
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        admin = verify_admin_login(email, password)

        if admin:
            session["admin"] = True
            session["admin_name"] = admin["name"]
            session["admin_email"] = admin["email"]
            return redirect(url_for("admin_dashboard"))

        # optional hard-coded fallback
        if email == "admin@caashmora.ac.in" and password == "admin@123":
            session["admin"] = True
            session["admin_name"] = "System Administrator"
            session["admin_email"] = email
            return redirect(url_for("admin_dashboard"))

        flash("❌ Invalid Admin Email or Password", "error")

    # reuse same Login.html for admin
    return render_template("Login.html")


# ---------------- ADMIN DASHBOARD ----------------
@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    # ---------- DATASET STATS & TRAINING ACCURACY ----------
    total_queries = 0
    total_intents = 0
    accuracy = "N/A"

    try:
        # Always read from CSV so stats update without restart
        if os.path.exists(TRAINING_FILE):
            df = pd.read_csv(TRAINING_FILE, encoding="latin1")

            total_queries = len(df)
            total_intents = df["intent"].astype(str).nunique()

            # If model loaded, compute training accuracy
            if hasattr(bot, "model") and bot.model is not None:
                X = df["text"].astype(str)
                y = df["intent"].astype(str)
                preds = bot.model.predict(X)
                acc_value = (preds == y).mean() * 100.0
                accuracy = f"{acc_value:.1f}%"
    except Exception as e:
        print("DASHBOARD DATASET / ACCURACY ERROR:", e)

    # ---------- LAST RETRAINED TIME ----------
    lr_path = os.path.join(BASE_DIR, "last_retrained.txt")
    if os.path.exists(lr_path):
        try:
            with open(lr_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if raw:
                dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
                last_retrained = dt.strftime("%d-%m-%Y %I:%M:%S %p")
            else:
                last_retrained = "Not retrained yet"
        except Exception as e:
            print("LAST_RETRAIN READ ERROR:", e)
            last_retrained = "Not retrained yet"
    else:
        last_retrained = "Not retrained yet"

    # ---------- RECENT USER QUERIES TABLE ----------
    recent = get_recent_chats(limit=5)
    formatted = []
    for r in recent:
        conf = r["confidence"]
        if conf is None:
            conf_str = "—%"
        else:
            val = float(conf)
            if 0 <= val <= 1:
                val *= 100.0          # 0.7 -> 70.0
            conf_str = f"{val:.1f}%"

        formatted.append(
            {
                "account": r["account"],
                "user_message": r["user_message"],
                "intent": r["intent"],
                "confidence": conf_str,
                "timestamp": r["timestamp"],
            }
        )

    return render_template(
        "admin_dashboard.html",
        total_queries=total_queries,
        total_intents=total_intents,
        accuracy=accuracy,
        last_retrained=last_retrained,
        recent_queries=formatted,
        chatlogs_url=url_for("admin_chatlogs"),
    )


# ---------------- ADMIN: QUERIES ----------------
@app.route("/admin_queries")
def admin_queries():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        SELECT account, user_message, bot_response, intent, confidence, timestamp
        FROM chat_logs
        ORDER BY id DESC
    """
    )
    rows = c.fetchall()
    conn.close()

    return render_template("admin_queries.html", queries=rows)


# ---------------- ADMIN: AUTO FAQ (from chat logs) ----------------
@app.route("/admin_faq")
def admin_faq():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    faqs = get_frequent_questions()
    return render_template("admin_faq.html", faqs=faqs)


# ---------------- ADMIN TRAINING ----------------
def append_training_sample(text, intent, response):
    file_exists = os.path.exists(TRAINING_FILE)
    with open(TRAINING_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "intent", "response"])
        writer.writerow([text, intent, response])


@app.route("/admin_training", methods=["GET", "POST"])
def admin_training():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        intent = request.form.get("intent", "").strip()
        response = request.form.get("response", "").strip()

        if text and intent and response:
            append_training_sample(text, intent, response)
            flash("✅ Training example added to dataset.", "success")
        else:
            flash("⚠️ Please fill all fields (text, intent, response).", "error")

        return redirect(url_for("admin_training"))

    # For UI: show frequent questions
    faqs = get_frequent_questions()

    # All existing intents from current ML dataset (so admin can reuse)
    try:
        intent_values = sorted(set(str(i) for i in bot.df["intent"].unique()))
    except Exception:
        intent_values = []

    # Load full training dataset from CSV to display
    training_data = []
    try:
        if os.path.exists(TRAINING_FILE):
            df_view = pd.read_csv(TRAINING_FILE, encoding="latin1")
            training_data = df_view.to_dict(orient="records")
    except Exception as e:
        print("TRAINING DATA READ ERROR:", e)

    return render_template(
        "admin_training.html",
        faqs=faqs,
        intents=intent_values,
        training_data=training_data,
    )


# ---------------- ADMIN: RETRAIN MODEL ----------------
# ---------------- ADMIN: RETRAIN MODEL ----------------
@app.route("/admin_retrain", methods=["POST"])
def admin_retrain():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    try:
        global bot
        bot = importlib.reload(bot)
        reset_all_bot_context()

        # Save retrain time to file
        lr_path = os.path.join(BASE_DIR, "last_retrained.txt")
        with open(lr_path, "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        flash("✅ Model retrained from latest training data.", "success")
    except Exception as e:
        print("RETRAIN ERROR:", e)
        flash("❌ Failed to retrain model. Check server logs.", "error")

    return redirect(url_for("admin_dashboard"))


# ---------------- ADMIN CHAT LOGS ----------------
@app.route("/admin_chatlogs")
def admin_chatlogs():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT user_message, bot_response, timestamp
        FROM chat_logs
        ORDER BY id DESC
    """)
    rows = c.fetchall()
    conn.close()

    formatted = []
    for r in rows:
        t = r["timestamp"]
        try:
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            t = (dt + timedelta(hours=5, minutes=30)).strftime("%d %b %Y - %I:%M %p")
        except Exception:
            pass
        formatted.append((r["user_message"], r["bot_response"], t))

    return render_template("chat_logs.html", logs=formatted, account="ADMIN")


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


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    create_db()
    ensure_columns()
    app.run(debug=True, port=5000)
