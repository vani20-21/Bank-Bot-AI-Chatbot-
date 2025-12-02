import pandas as pd
import re
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from db import record_transaction
import sqlite3

DB_PATH = "bank.db"

def get_balance(account):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT balance FROM users WHERE account_number=?", (account,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def update_balance(account, new_balance):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE users SET balance=? WHERE account_number=?", (new_balance, account))
    conn.commit()
    conn.close()

def random_txn_id():
    import random, string
    return "TXN" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
   

# ========= Config =========
DATA_FILE = "bankbot_final_expanded1.csv"
CONFIDENCE_THRESHOLD = 0.55
ANNUAL_RATE = 0.085  # 8.5% annually

# ========= Utils =========
def normalize_text(s): 
    return s.strip().lower()

def random_txn_id(): 
    return "TXN" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

def random_balance(): 
    return f"₹{random.randint(1000, 500000):,}"

def mask_aadhaar(a):
    a = re.sub(r'\D', '', a)
    if len(a) >= 4:
        return "**** **** " + a[-4:]
    return "****"

def calculate_emi(P: float, r: float, n: int):
    if n <= 0:
        raise ValueError("n must be > 0")
    return (P * r * (1 + r)**n) / ((1 + r)**n - 1)

# ========= Entity Extraction =========
def extract_entities(text):
    ent = {}
    raw = text.strip()

    # last-4 (pure 4 digits token)
    m4 = re.fullmatch(r'\s*(\d{4})\s*', raw)
    if m4:
        ent['last4'] = m4.group(1)

    # account number 6-16 digits
    acc = re.search(r'\b\d{6,16}\b', raw)
    if acc:
        ent['account_number'] = acc.group()

    # money (₹ / Rs / INR or bare number)
    m = re.search(r'(?:₹\s?|rs\.?\s?|inr\s?|\b)(\d{1,12}(?:,\d{3})*(?:\.\d{1,2})?)', raw, re.I)
    if m:
        ent['money'] = m.group(1).replace(',', '')
    else:
        onlynum = re.fullmatch(r'\s*([0-9]{2,12})\s*', raw)
        if onlynum:
            ent['money'] = onlynum.group(1)

    # payment method
    if re.search(r'\bupi\b', raw, re.I):
        ent['payment_method'] = 'UPI'
    elif re.search(r'\b(bank transfer|neft|imps|rtgs)\b', raw, re.I):
        ent['payment_method'] = 'Bank Transfer'

    # receiver name
    nm = re.search(r'(?:to|pay|send|transfer to)\s+([A-Za-z][A-Za-z.\' \-]{1,40})', raw, re.I)
    if nm:
        ent['receiver_name'] = nm.group(1).strip().title()

    return ent

# ========= Load model (optional dataset replies) =========
try:
    df = pd.read_csv(DATA_FILE, encoding='latin1')
    has_data = True
except FileNotFoundError:
    df = pd.DataFrame(columns=["text", "intent", "response"])
    has_data = False

if has_data and not all(col in df.columns for col in ["text", "intent", "response"]):
    raise SystemExit("CSV must contain columns: text,intent,response")

if has_data:
    X = df["text"].astype(str)
    y = df["intent"].astype(str)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=18000),
        LogisticRegression(max_iter=2500)
    )
    model.fit(X, y)
else:
    model = None

# ========= Dataset helper (safe) =========
def dataset_response_for_intent(intent, user_input):
    if not has_data:
        return None
    subset = df[df["intent"] == intent]
    if subset.empty:
        return None
    exact = subset[subset["text"].str.strip().str.lower() == user_input.strip().lower()]
    if not exact.empty:
        return exact.iloc[0]["response"]
    return subset.sample(1, random_state=random.randint(1, 1_000_000)).iloc[0]["response"]

# ========= Memory =========
memory = {
    # high-level router
    "menu": None,  # 'card','atm','loan','account'
    # card flow
    "card": {
        "type": None,   # 'debit'|'credit'
        "action": None, # 'block','unblock','status','apply','report','viewbill','paybill'
        "step": 0,
        "last4": None,
        "amount": None
    },
    # atm flow
    "atm": {
        "submenu": False,
        "action": None,  # 'locator','limit','issue','not_dispensed','pin_change'
        "last4": None
    },
    # loan flow
    "loan": {
        "category": None,  # 'secured','unsecured','business'
        "product": None,   # 'home','auto','lap','gold','fd' | 'personal','education','credit','debt' | 'term','wc','equip','invoice','od'
        "service": None,   # 'elig','apply','status'
        "elig": { "age":None, "salary":None, "emp":None, "exp":None, "cibil":None, "eligible":None, "amount":None },
        "apply": { "name":None, "salary":None, "pan":None, "uploaded":False },
        "waiting_apply": False,
        "step": 0,
        "must_check_first": False
    },
    # account open
    "acct": {
        "step": 0,
        "name": None,
        "age": None,
        "type": None,  # 'savings'|'current'
        "addr": None,
        "aadhaar": None
    },
    # transfer/balance
    "flow": None, "step": 0, "receiver":None, "account":None, "amount":None,
    "last_intent": None
}

def reset_card():
    memory["card"] = {"type":None,"action":None,"step":0,"last4":None,"amount":None}

def reset_atm():
    memory["atm"] = {"submenu":False,"action":None,"last4":None}

def reset_loan():
    memory["loan"] = {
        "category": None,
        "product": None,
        "service": None,
        "step": 0,
        "must_check_first": False,

        "elig": {
            "age": None,
            "salary": None,
            "emp": None,
            "exp": None,
            "cibil": None,
            "eligible": None,
            "amount": None
        },

        # ⭐ THIS PART WAS MISSING OR BROKEN ⭐
        "apply": {
            "name": None,
            "salary": None,
            "pan": None,
            "business_name": None,   # <--- for business loans
            "gst": None,            # <--- for business loans
            "uploaded": False
        },

        "waiting_apply": False
    }

def reset_acct():
    memory["acct"] = {"step":0,"name":None,"age":None,"type":None,"addr":None,"aadhaar":None}

def clear_transfer_memory():
    for k in ["flow","step","receiver","receiver_account","amount"]:
        memory[k] = None if k != "step" else 0

# ========= Quick helpers =========
def is_yes(s): 
    return bool(re.search(r'\b(yes|y|sure|continue|proceed)\b', normalize_text(s)))

def wants_debit(s):
    t = normalize_text(s)
    return bool(re.search(r'\bdebit( card)?\b', t)) or s.strip()=="1"

def wants_credit(s):
    t = normalize_text(s)
    return bool(re.search(r'\bcredit( card)?\b', t)) or s.strip()=="2"

def is_number_choice(s, low, high):
    return s.strip().isdigit() and low <= int(s.strip()) <= high

# ====== Menus (texts) ======
CARD_ASK = "Would you like Debit Card or Credit Card services?\n1) Debit Card\n2) Credit Card"
DEBIT_MENU = ("Debit Card Services:\n"
              "1) Block Debit Card\n"
              "2) Unblock Debit Card\n"
              "3) Check Debit Card Status\n"
              "4) Apply for a New Debit Card\n"
              "5) Report Lost/Stolen Card")
CREDIT_MENU = ("Credit Card Services:\n"
               "1) Block Credit Card\n"
               "2) Unblock Credit Card\n"
               "3) Check Credit Card Status\n"
               "4) Apply for a New Credit Card\n"
               "5) View Bill/Payment\n"
               "6) Pay Bill")
ATM_MENU = ("ATM Services:\n"
            "1) Locate Nearest ATM\n"
            "2) ATM Withdrawal Limit\n"
            "3) Report ATM Cash Withdrawal Issue\n"
            "4) Card Not Dispensed\n"
            "5) Change ATM PIN")

LOAN_MAIN = "Please choose a loan category:\n1) Secured Loan\n2) Unsecured Loan\n3) Business Loan"
SECURED_MENU = "Secured Loan Available:\n1) Home Loan\n2) Auto / Vehicle Loan\n3) Loan Against Property\n4) Gold Loan\n5) Loan Against Fixed Deposit"
UNSEC_MENU = "Unsecured Loan Available:\n1) Personal Loan\n2) Education Loan\n3) Credit Card (Revolving Loan)\n4) Debt Consolidation Loan"
BUS_MENU = "Business Loans Available:\n1) Term Loan\n2) Working Capital Loan\n3) Equipment Financing\n4) Invoice Financing\n5) Overdraft Facility"
LOAN_SERVICE = "What would you like to do?\n1) Check Eligibility\n2) Apply for Loan\n3) Check Application Status"

DOCS_SECURED = (
    "📄 Required Documents for Secured Loans (Home/Auto/LAP/Gold/FD):\n\n"
    "• Identity Proof: Aadhaar / PAN / Passport / Voter ID\n"
    "• Address Proof: Electricity Bill / Rental Agreement / Aadhaar\n"
    "• Income Proof: Salary Slips (3 months) OR ITR (2 years for self-employed)\n"
    "• Bank Statement: Last 6 months\n"
    "• Property / Asset Documents (Sale deed / valuation report) — for Home & LAP\n"
    "• Vehicle Quotation / Invoice — for Auto Loan\n"
    "• Gold to be brought to branch for purity verification — for Gold Loan\n"
    "• FD receipt — for FD Loan\n"
)

DOCS_UNSECURED = (
    "📄 Required Documents for Unsecured Loans (Personal / Education / Credit / Debt Consolidation):\n\n"
    "• Identity Proof: Aadhaar / PAN / Passport / Voter ID\n"
    "• Address Proof: Electricity Bill / Rental Agreement / Aadhaar\n"
    "• Income Proof: Salary Slips (3 months) OR ITR (2 years for self-employed)\n"
    "• Bank Statement: Last 6 months\n"
    "• CIBIL Score must meet product requirements\n"
    "• For Education Loan: Admission Proof + Fee Structure + Co-Applicant KYC\n"
)

DOCS_BUSINESS = (
    "📄 Required Documents for Business Loans (Term / Working Capital / Equipment / Invoice / OD):\n\n"
    "• Business KYC: GST Certificate + Udyam Registration\n"
    "• Identity Proof of Promoter: Aadhaar / PAN\n"
    "• Address Proof of Business & Promoter\n"
    "• ITR for last 2 years\n"
    "• Bank Statement for last 12 months\n"
    "• Business Financials: Balance Sheet & Profit/Loss Statement\n"
    "• Business Registration Proof (Partnership Deed / MSME Cert / Incorporation Cert)\n"
    "• For Equipment Loan: Equipment Invoice/Quotation\n"
    "• For Invoice Financing: Invoice Copy + Buyer Details\n"
)

# ========= Core handler =========
def handle_user_input(user_input):
    raw = user_input.strip()
    text = normalize_text(raw)
    ent = extract_entities(raw)


    # --- after eligibility decision (apply or not now) ---
    if memory.get("loan", {}).get("waiting_apply"):
        if "apply" in text:
            memory["loan"]["waiting_apply"] = False
            L = memory["loan"]
            L["service"] = "apply"
            L["step"] = 10
            return "loan_apply_start", {}, "Please provide your full name."

        if text in ["no", "not now", "later", "stop", "cancel"]:
            reset_loan(); memory["menu"] = None
            return "reject", {}, "No problem, I'm here whenever you're ready."

        return "loan_eligibility_result", {}, "Type 'apply' to continue or 'not now' to cancel."


    # prevent menu number from being treated as money
    if memory.get("menu") and raw.strip() in ["1","2","3","4","5","6"]:
        ent.pop("money", None)

    # ===== greetings
    if re.search(r'\b(hi|hello|hey)\b', text):
        return "greet", ent, "Hello, how may I assist you?"

    # ===== quick openers for card types (text OR number)
    # user can say "card", "debit card", "credit card" directly
    if text in ["card","cards"]:
        memory["menu"]="card"; reset_card()
        return "card_menu", ent, CARD_ASK

    if re.search(r'\bdebit( card)?\b', text):
        memory["menu"]="card"; reset_card()
        memory["card"]["type"]="debit"
        return "debit_menu", ent, DEBIT_MENU

    if re.search(r'\bcredit( card)?\b', text):
        memory["menu"]="card"; reset_card()
        memory["card"]["type"]="credit"
        return "credit_menu", ent, CREDIT_MENU

    # ===== balance check
    if re.search(r'\b(balance|check balance|account balance)\b', text):
        memory["last_intent"] = "balance"
        return "balance_enquiry", ent, "Please provide your account number to view the balance."

    if memory.get("last_intent") == "balance" and re.fullmatch(r'\d{6,16}', raw):
        memory["last_intent"] = None

        from db import get_balance  # use db version if moved

        balance = get_balance(raw)
        if balance is None:
            return "check_balance", {}, "❌ Account not found in records."

        # ❗ DO NOT set memory["current_user_account"] here
        # Logged-in account is already stored in session → Flask injects into bot.

        return "check_balance", {"account": raw}, f"The current balance for account {raw} is ₹{balance:,}."


    # ================= MONEY TRANSFER (DB CONNECTED) =================
    if re.search(r'\b(pay|transfer|send)\b', text) and memory.get("flow") != "transfer":
        memory["flow"] = "transfer"
        memory["step"] = 1
        memory["receiver_name"] = None
        memory["receiver_account"] = None
        memory["amount"] = None
        return "transfer_money", {}, "To whom would you like to transfer money?"

    if memory.get("flow") == "transfer":

        # STEP 1: Receiver Name
        if memory["step"] == 1:
            memory["receiver_name"] = raw.strip().title()
            memory["step"] = 2
            return "transfer_money", {}, f"Please provide {memory['receiver_name']}'s account number."

        # STEP 2: Receiver Account Number
        if memory["step"] == 2:
            acc = re.sub(r'\D', '', raw)
            if not re.fullmatch(r'\d{6,16}', acc):
                return "transfer_money", {}, "Please enter a valid account number (6–16 digits)."
            memory["receiver_account"] = acc
            memory["step"] = 3
            return "transfer_money", {}, "Please enter the amount to transfer."

        # STEP 3: Amount
        if memory["step"] == 3:
            amt = re.search(r'\d+', raw.replace(",", ""))
            if not amt:
                return "transfer_money", {}, "Please enter amount in digits only."
            memory["amount"] = int(amt.group())
            memory["step"] = 4
            return "transfer_money", {}, f"Please choose payment method to transfer ₹{memory['amount']}: UPI or Bank Transfer?"

        # STEP 4: Payment Method + PROCESS TRANSFER
        if memory["step"] == 4:
            if "upi" in text:
                pm = "UPI"
            elif "bank" in text or "neft" in text or "rtgs" in text or "imps" in text:
                pm = "Bank Transfer"
            else:
                return "transfer_money", {}, "Invalid payment method. Type 'UPI' or 'Bank Transfer'."

            from db import get_balance, update_balance, get_user_by_account, record_transaction
            import random, string

            sender_acct = memory.get("current_user_account")
            receiver_acct = memory.get("receiver_account")
            receiver_name = memory.get("receiver_name")
            amount = memory.get("amount")

            # ❌ Prevent sending to self
            if sender_acct == receiver_acct:
                memory["flow"] = None
                return "transfer_money", {}, "❌ You cannot transfer money to your own account."

            # ✅ Check balance
            sender_balance = get_balance(sender_acct)
            if sender_balance is None or sender_balance < amount:
                memory["flow"] = None
                return "transfer_money", {}, "❌ Transaction Failed: Insufficient Balance."

            # ✅ Deduct/credit balances
            update_balance(sender_acct, sender_balance - amount)

            receiver = get_user_by_account(receiver_acct)
            if receiver:
                update_balance(receiver_acct, receiver["balance"] + amount)

            # ✅ Save transaction log
            record_transaction(sender_acct, receiver_acct, receiver_name, amount, pm, "Success")

            txn = "TXN" + ''.join(random.choices(string.digits, k=6))

            memory["flow"] = None
            return "transfer_money", {}, (
                f"✅ Transfer Successful!\n"
                f"₹{amount:,} transferred to {receiver_name} (A/C: {receiver_acct}) via {pm}.\n"
                f"Transaction ID: {txn}"
            )



        # ===== CARD FLOW (fixed: numbers + text, deterministic replies)
        if memory.get("menu")=="card":
            c = memory["card"]

            # choose type
            if c["type"] is None:
                if wants_debit(raw):
                    c["type"]="debit"
                    return "debit_menu", {}, DEBIT_MENU
                if wants_credit(raw):
                    c["type"]="credit"
                    return "credit_menu", {}, CREDIT_MENU
                return "card_menu", {}, "Please choose 1 for Debit or 2 for Credit."

            # ---- Debit actions
            if c["type"]=="debit":
                if c["action"] is None:
                    # by number
                    if is_number_choice(raw,1,5):
                        c["action"] = {"1":"block","2":"unblock","3":"status","4":"apply","5":"report"}[raw.strip()]
                    else:
                        t = text
                        if re.search(r'\bblock\b', t): c["action"]="block"
                        elif re.search(r'\bunblock\b', t): c["action"]="unblock"
                        elif re.search(r'\b(status|check)\b', t): c["action"]="status"
                        elif re.search(r'\bapply\b', t): c["action"]="apply"
                        elif re.search(r'\breport|lost|stolen\b', t): c["action"]="report"
                    if c["action"] is None:
                        return "debit_menu", {}, DEBIT_MENU

                # apply does NOT need last4
                if c["action"]=="apply":
                    reset_card(); memory["menu"]=None
                    return "debit_card_replacement", {}, "Your Debit Card request has been submitted successfully.\nFurther application details will be sent to your registered mobile number and email.\n\nWould you like to continue?"

                # other actions need last4
                if not c["last4"]:
                    if 'last4' in ent and re.fullmatch(r'\d{4}', ent['last4']):
                        c["last4"]=ent['last4']
                    else:
                        return "ask_card_last4", {}, "For security, please enter the last 4 digits of your debit card."

                # perform
                last4 = c["last4"]
                action = c["action"]
                reset_card(); memory["menu"]=None

                if action=="block":
                    return "debit_card_block", {}, f"Your Debit Card ending with **** **** **** {last4} has been blocked successfully.\n\nWould you like to continue?"
                if action=="unblock":
                    return "debit_card_unblock", {}, f"Your Debit Card ending with **** **** **** {last4} has been unblocked.\n\nWould you like to continue?"
                if action=="status":
                    return "debit_card_status", {}, f"Your Debit Card **** **** **** {last4} is active.\n\nWould you like to continue?"
                if action=="report":
                    return "debit_card_report_lost", {}, f"Lost/Stolen report filed. Debit Card **** **** **** {last4} is now blocked.\n\nWould you like to continue?"

            # ---- Credit actions
            if c["type"]=="credit":
                if c["action"] is None:
                    if is_number_choice(raw,1,6):
                        c["action"] = {"1":"block","2":"unblock","3":"status","4":"apply","5":"viewbill","6":"paybill"}[raw.strip()]
                    else:
                        t = text
                        if re.search(r'\bblock\b', t): c["action"]="block"
                        elif re.search(r'\bunblock\b', t): c["action"]="unblock"
                        elif re.search(r'\b(status|limit|check)\b', t): c["action"]="status"
                        elif re.search(r'\bapply\b', t): c["action"]="apply"
                        elif re.search(r'\bpay bill|bill payment|payment\b', t): c["action"]="paybill"
                        elif re.search(r'\bview bill|bill\b', t): c["action"]="viewbill"
                    if c["action"] is None:
                        return "credit_menu", {}, CREDIT_MENU

                if c["action"]=="apply":
                    reset_card(); memory["menu"]=None
                    return "credit_card_application", {}, "Your Credit Card request has been submitted successfully.\nFurther application details will be sent to your registered mobile number and email.\n\nWould you like to continue?"

                # other credit actions need last4
                if not c["last4"]:
                    if 'last4' in ent and re.fullmatch(r'\d{4}', ent['last4']):
                        c["last4"]=ent['last4']
                    else:
                        return "ask_card_last4", {}, "For security, please enter the last 4 digits of your credit card."

                last4 = c["last4"]
                action = c["action"]

                # paybill needs amount
                if action=="paybill" and not c["amount"]:
                    if 'money' in ent:
                        try:
                            c["amount"] = str(int(float(ent['money'])))
                        except:
                            c["amount"] = None
                    if not c["amount"]:
                        return "credit_card_payment", {}, "Enter bill amount to pay (e.g., 2500)."

                # finalize actions
                msg = ""
                if action=="block":
                    msg = f"Your Credit Card **** **** **** {last4} has been blocked successfully."
                elif action=="unblock":
                    msg = f"Your Credit Card **** **** **** {last4} has been unblocked."
                elif action=="status":
                    msg = f"Your Credit Card **** **** **** {last4} is active. Limit changes require KYC."
                elif action=="viewbill":
                    msg = f"Latest bill for **** **** **** {last4} is available in your statements."
                elif action=="paybill":
                    amt = int(float(c["amount"]))
                    msg = f"Payment of ₹{amt:,} received for Credit Card **** **** **** {last4}."

                reset_card(); memory["menu"]=None
                return "credit_card_action", {}, msg + "\n\nWould you like to continue?"

	    # ===== ATM FLOW (keep same behavior)
    if text in ["atm","atms"]:
        memory["menu"]="atm"; reset_atm()
        return "atm_menu", {}, ATM_MENU

    if memory.get("menu")=="atm":
        a = memory["atm"]
        if a["action"] is None:
            # choose by number or text
            if is_number_choice(raw,1,5):
                a["action"] = { "1":"locator","2":"limit","3":"issue","4":"not_dispensed","5":"pin_change" }[raw.strip()]
            else:
                t = text
                if re.search(r'nearest|locate|near',t): a["action"]="locator"
                elif re.search(r'limit|withdrawal',t): a["action"]="limit"
                elif re.search(r'issue|problem',t): a["action"]="issue"
                elif re.search(r'not dispensed|card not dispensed',t): a["action"]="not_dispensed"
                elif re.search(r'pin',t): a["action"]="pin_change"
            if a["action"] is None: 
                return "atm_menu", {}, ATM_MENU

        # locator does not need last4
        if a["action"]=="locator":
            reset_atm(); memory["menu"]=None
            return "atm_locator", {}, "Nearest ATMs will be shown based on your location.\n\nWould you like to continue?"

        # others require last4 first
        if not a["last4"]:
            if 'last4' in ent and re.fullmatch(r'\d{4}', ent['last4']):
                a["last4"]=ent['last4']
            else:
                return "ask_card_last4", {}, "Please enter the last 4 digits of your card to proceed."

        if a["action"]=="limit":
            reset_atm(); memory["menu"]=None
            return "atm_withdrawal_limit", {}, "ATM withdrawal limit is ₹40,000/day.\n\nWould you like to continue?"
        if a["action"]=="issue":
            reset_atm(); memory["menu"]=None
            return "atm_issue_report", {}, "ATM issue reported. Resolution within 48 hours.\n\nWould you like to continue?"
        if a["action"]=="not_dispensed":
            reset_atm(); memory["menu"]=None
            return "atm_card_not_dispensed", {}, "Card not dispensed — amount will be auto-reversed if debited.\n\nWould you like to continue?"
        if a["action"]=="pin_change":
            reset_atm(); memory["menu"]=None
            return "atm_pin_change", {}, "Change your ATM PIN using the mobile app or ATM.\n\nWould you like to continue?"

    # ===== LOAN FLOW
    if text in ["loan","loans"]:
        memory["menu"]="loan"; reset_loan()
        return "loan_menu", {}, LOAN_MAIN

    if memory.get("menu")=="loan":
        L = memory["loan"]

        # Step 1: category
        if L["category"] is None:
            if raw.strip()=="1" or "secured" in text:
                L["category"]="secured"; return "loan_type_menu", {}, SECURED_MENU
            if raw.strip()=="2" or "unsecured" in text:
                L["category"]="unsecured"; return "loan_type_menu", {}, UNSEC_MENU
            if raw.strip()=="3" or "business" in text:
                L["category"]="business"; return "loan_type_menu", {}, BUS_MENU
            return "loan_menu", {}, LOAN_MAIN

        # Step 2: product by category
        if L["product"] is None:
            t = raw.strip()
            if L["category"]=="secured":
                mapping = {"1":"home","2":"auto","3":"lap","4":"gold","5":"fd"}
                textmap = [("home","home"),("auto","auto"),("vehicle","auto"),
                           ("property","lap"),("gold","gold"),("fixed deposit","fd")]
                if t in mapping: L["product"]=mapping[t]
                else:
                    for k,v in textmap:
                        if k in text: L["product"]=v
                if L["product"] is None: return "loan_type_menu", {}, SECURED_MENU
            elif L["category"]=="unsecured":
                mapping = {"1":"personal","2":"education","3":"credit","4":"debt"}
                textmap = [("personal","personal"),("education","education"),
                           ("credit card","credit"),("revolving","credit"),("debt","debt")]
                if t in mapping: L["product"]=mapping[t]
                else:
                    for k,v in textmap:
                        if k in text: L["product"]=v
                if L["product"] is None: return "loan_type_menu", {}, UNSEC_MENU
            else:
                mapping = {"1":"term","2":"wc","3":"equip","4":"invoice","5":"od"}
                textmap = [("term","term"),("working","wc"),("equipment","equip"),
                           ("invoice","invoice"),("overdraft","od")]
                if t in mapping: L["product"]=mapping[t]
                else:
                    for k,v in textmap:
                        if k in text: L["product"]=v
                if L["product"] is None: return "loan_type_menu", {}, BUS_MENU

            return "loan_service_menu", {}, LOAN_SERVICE

        # Step 3: service
        if L["service"] is None:
            if raw.strip()=="1" or re.search(r'eligib', text):
                L["service"]="elig"; L["step"]=1
                return "loan_eligibility_check", {}, "Please enter your age in years."
            elif raw.strip()=="2" or re.search(r'\bapply\b', text):
                # force eligibility first
                L["service"]="elig"; L["step"]=1
                return "loan_eligibility_required", {}, "Please check eligibility first. Enter your age in years."
            elif raw.strip()=="3" or re.search(r'status', text):
                L["service"]="status"; L["step"]=100
                return "loan_status", {}, "Please enter your application number."
            else:
                return "loan_service_menu", {}, LOAN_SERVICE

        # Status
        if L["service"]=="status" and L["step"]==100:
            if not re.fullmatch(r'APP[0-9A-Z]{8,}', raw.strip(), re.I):
                return "loan_status", {}, "Please enter a valid application number (e.g., APP12345678)."
            reset_loan(); memory["menu"]=None
            return "loan_status_result", {}, "Your application is under review. You will be notified by SMS/Email."

        # Eligibility
        if L["service"]=="elig":
            E = L["elig"]

            # ===== EDUCATION LOAN SPECIAL RULES =====
            if L["product"] == "education":
                # Step 1: student age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter student age."
                    E["age"] = int(raw.strip())
                    if E["age"] < 17:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Student must be at least 17 years old."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "Enter parent's monthly income."

                # Step 2: parent income (>=25,000)
                if L["step"] == 2:
                    m = re.search(r'(\d{4,9})', raw.replace(',',''))
                    if not m:
                        return "loan_eligibility_check", {}, "Please enter numeric income. Example: 30000"
                    E["salary"] = int(m.group(1))
                    if E["salary"] < 15000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Parent income must be at least ₹15,000/month."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Parent employment type?\n1) Government\n2) Private"

                # Step 3: parent employment
                if L["step"] == 3:
                    if raw.strip() == "1":
                        E["emp"] = "govt"
                    elif raw.strip() == "2":
                        E["emp"] = "private"
                    else:
                        return "loan_eligibility_check", {}, "Please type 1 for Government or 2 for Private."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter parent CIBIL score."

                # Step 4: parent CIBIL (>=700)
                if L["step"] == 4:

                    # allow: "what is cibil", "explain cibil", etc.
                    if re.search(r'what.*cibil|cibil meaning|explain cibil', text):
                        return "loan_info", {}, (
                            "CIBIL score shows credit repayment history.\n"
                            "Range: 300–900. Above 750 is considered good.\n\n"
                            "Please enter parent CIBIL score (300–900)."
                        )

                    if not re.fullmatch(r'\d{3,4}', raw.strip()):
                        return "loan_eligibility_check", {}, "Enter valid CIBIL (300–900)."

                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 700:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: CIBIL score must be 700 or above."

                    L["step"] = 5
                    return "loan_eligibility_check", {}, "Is the course in 'India' or 'Abroad'?"

                # Step 5: course location
                if L["step"] == 5:
                    if "india" in text:
                        E["location"] = "India"
                    elif "abroad" in text:
                        E["location"] = "Abroad"
                    else:
                        return "loan_eligibility_check", {}, "Please type 'India' or 'Abroad'."
                    L["step"] = 6
                    return "loan_eligibility_check", {}, "How much loan amount do you need?"

                # Step 6: requested amount + collateral rules + proceed to apply
                if L["step"] == 6:
                    amt = re.search(r'\d{4,9}', raw.replace(',',''))
                    if not amt:
                        return "loan_eligibility_check", {}, "Please enter amount numeric. Example: 600000"
                    E["amount"] = int(amt.group(0))

                    if E["amount"] <= 400000:
                        collateral = "No collateral required (Parent will be co-applicant)"
                    elif E["amount"] <= 750000:
                        collateral = "Third party guarantee required"
                    else:
                        collateral = "Tangible collateral (Property / Fixed Deposit) required"

                    E["eligible"] = True
                    # hand off to apply flow like other loans
                    L["service"] = "apply"
                    L["step"] = 10  # next prompt asks for full name
                    memory["loan"]["waiting_apply"] = True

                    return (
                        "loan_eligibility_result",
                        {},
                        f"Education Loan Eligibility Result\n\n"
                        f"Loan Amount Considered: ₹{E['amount']:,}\n"
                        f"Course Location: {E['location']}\n"
                        f"Parent Income: ₹{E['salary']:,}/month\n"
                        f"CIBIL Score: {E['cibil']} (Good)\n\n"
                        f"Collateral: {collateral}\n"
                        f"Co-Applicant: Parent/Guardian mandatory\n"
                        f"Repayment: After course + job placement\n\n"
                        f"Type 'apply' to continue with application."
                    )
                
            # ===== PERSONAL LOAN ELIGIBILITY (Central Bank of India Rules) =====
            if L["product"] == "personal":
                # Step 1 — Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter your age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 21:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum age required is 21."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "Enter your monthly income (₹)."

                # Step 2 — Income (Min ₹15,000 / month)
                if L["step"] == 2:
                    m = re.search(r'(\d{4,9})', raw.replace(',',''))
                    if not m:
                        return "loan_eligibility_check", {}, "Please enter numeric income, e.g., 25000."
                    E["salary"] = int(m.group(1))
                    if E["salary"] < 15000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum income required is ₹15,000/month."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, (
                        "Select Employment Type:\n"
                        "1) Government Employee\n"
                        "2) Private/MNC Employee\n"
                        "3) Self-Employed"
                    )

                # Step 3 — Employment Type
                if L["step"] == 3:
                    choice = raw.strip()
                    if choice == "1":
                        E["emp"] = "govt"
                    elif choice == "2":
                        E["emp"] = "private"
                    elif choice == "3":
                        E["emp"] = "self"
                    else:
                        return "loan_eligibility_check", {}, "Please choose 1, 2, or 3."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter total years of work experience."

                # Step 4 — Experience Rules
                if L["step"] == 4:
                    if not re.fullmatch(r'\d{1,2}', raw.strip()):
                        return "loan_eligibility_check", {}, "Please enter years as a number."
                    E["exp"] = int(raw.strip())

                    if E["emp"] == "govt" and E["exp"] < 1:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Government employees need at least 1 year of service."
                    if E["emp"] == "private" and E["exp"] < 3:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Private employees need minimum 3 years service."
                    if E["emp"] == "self" and E["exp"] < 2:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Self-employed requires minimum 2 years business proof."

                    L["step"] = 5
                    return "loan_eligibility_check", {}, "Enter your CIBIL score (300–900)."

                # Step 5 — CIBIL Score
                if L["step"] == 5:

                    if re.search(r'what.*cibil|cibil meaning|explain cibil', text):
                        return "loan_info", {}, (
                            "CIBIL score shows how you repaid past loans.\n"
                            "Range: 300–900. Above 750 is ideal.\n\n"
                            "Please enter CIBIL score (300–900)."
                        )

                    if not re.fullmatch(r'\d{3,4}', raw.strip()):
                        return "loan_eligibility_check", {}, "Please enter a valid CIBIL score (300–900)."

                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 700:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: CIBIL score must be 700 or above."

                    max_salary_based = E["salary"] * 24
                    eligible_amount = min(max_salary_based, 2000000)  # 20L max

                    E["eligible"] = True
                    E["amount"] = eligible_amount
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    return "loan_eligibility_result", {}, (
                        f"Personal Loan Eligibility Confirmed.\n\n"
                        f"Eligible Loan Amount: ₹{eligible_amount:,}\n"
                        f"Tenure: Up to 7 years\n"
                        f"EMI will be calculated based on your income.\n\n"
                        f"Type 'apply' to continue or 'not now' to cancel."
                    )

            # ===== AUTO / VEHICLE LOAN RULES =====
            if L["product"] == "auto":
                # Step 1: Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter your age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 18:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum age is 18."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "Enter your monthly income (₹)."

                # Step 2: Income
                if L["step"] == 2:
                    m = re.search(r'(\d{4,9})', raw.replace(',',''))
                    if not m:
                        return "loan_eligibility_check", {}, "Please enter numeric income. Example: 30000"
                    E["salary"] = int(m.group(1))
                    if E["salary"] < 20000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum income must be ₹20,000/month."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Select Employment Type:\n1) Government Employee\n2) Private Employee\n3) Self-Employed"

                # Step 3: Employment Type
                if L["step"] == 3:
                    if raw.strip() == "1":
                        E["emp"] = "govt"
                    elif raw.strip() == "2":
                        E["emp"] = "private"
                    elif raw.strip() == "3":
                        E["emp"] = "self"
                    else:
                        return "loan_eligibility_check", {}, "Please select 1, 2, or 3."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter total years of work experience."

                # Step 4: Experience rules
                if L["step"] == 4:
                    if not re.fullmatch(r'\d{1,2}', raw.strip()):
                        return "loan_eligibility_check", {}, "Please enter experience in years (e.g., 2)."
                    E["exp"] = int(raw.strip())

                    if E["emp"] == "govt" and E["exp"] < 1:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Government employees need minimum 1 year experience."
                    if E["emp"] == "private" and E["exp"] < 3:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Private employees need minimum 3 years experience."
                    if E["emp"] == "self" and E["exp"] < 2:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Self-employed need minimum 2 years business proof."

                    L["step"] = 5
                    return "loan_eligibility_check", {}, "Enter your CIBIL score (300–900)."

                # Step 5: CIBIL Score
                if L["step"] == 5:
                    if not re.fullmatch(r'\d{3}', raw.strip()):
                        return "loan_eligibility_check", {}, "Enter valid CIBIL (300–900)."
                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 700:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: CIBIL score must be at least 700."
                    L["step"] = 6
                    return "loan_eligibility_check", {}, "Select Vehicle Type:\n1) Two-Wheeler\n2) Four-Wheeler (Car)"

                # Step 6: Vehicle Type
                if L["step"] == 6:
                    if raw.strip() == "1":
                        E["vehicle"] = "two"
                    elif raw.strip() == "2":
                        E["vehicle"] = "car"
                    else:
                        return "loan_eligibility_check", {}, "Please select 1 or 2."
                    L["step"] = 7
                    return "loan_eligibility_check", {}, "Enter the on-road vehicle price (₹)."

                # Step 7: Vehicle Price → Calculate LTV & Loan Amount
                if L["step"] == 7:
                    amt = re.search(r'\d{4,10}', raw.replace(',',''))
                    if not amt:
                        return "loan_eligibility_check", {}, "Please enter price numeric. Example: 85000"
                    price = int(amt.group(0))

                    if E["vehicle"] == "two":
                        loan_amt = int(price * 0.90)
                    else:  # car
                        if price <= 1200000:
                            loan_amt = int(price * 0.85)
                        else:
                            loan_amt = int(price * 0.80)

                    E["eligible"] = True
                    E["amount"] = loan_amt
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    return (
                        "loan_eligibility_result",
                        {},
                        f"Auto Loan Eligibility Confirmed.\n\n"
                        f"Vehicle Price: ₹{price:,}\n"
                        f"Eligible Loan Amount: ₹{loan_amt:,}\n"
                        f"Repayment Tenure: Up to 5–7 years based on vehicle type.\n\n"
                        f"Type 'apply' to continue or 'not now' to cancel."
                    )

            # ===== LOAN AGAINST PROPERTY (LAP) RULES =====
            if L["product"] == "lap":
                # Step 1: Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 21 or E["age"] > 70:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Age must be between 21 and 70."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "Enter your monthly income (₹)."

                # Step 2: Income requirement (>= 18,000/month)
                if L["step"] == 2:
                    m = re.search(r'(\d{4,9})', raw.replace(',',''))
                    if not m:
                        return "loan_eligibility_check", {}, "Please enter numeric income. Example: 22000"
                    E["salary"] = int(m.group(1))
                    if E["salary"] < 18000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum income must be ₹18,000/month."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Select Employment Type:\n1) Government Employee\n2) Private Employee\n3) Self-Employed"

                # Step 3: Employment Type
                if L["step"] == 3:
                    if raw.strip() == "1":
                        E["emp"] = "govt"
                    elif raw.strip() == "2":
                        E["emp"] = "private"
                    elif raw.strip() == "3":
                        E["emp"] = "self"
                    else:
                        return "loan_eligibility_check", {}, "Please select 1, 2, or 3."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter total years of work/business experience."

                # Step 4: Experience validation
                if L["step"] == 4:
                    if not re.fullmatch(r'\d{1,2}', raw.strip()):
                        return "loan_eligibility_check", {}, "Please enter experience in years (e.g., 2)."
                    E["exp"] = int(raw.strip())

                    if E["emp"] == "govt" and E["exp"] < 1:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Government employees need minimum 1 year experience."
                    if E["emp"] == "private" and E["exp"] < 3:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Private employees need minimum 3 years experience."
                    if E["emp"] == "self" and E["exp"] < 2:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Self-employed need minimum 2 years business proof."

                    L["step"] = 5
                    return "loan_eligibility_check", {}, "Enter your CIBIL score (300–900)."

                # Step 5: CIBIL
                if L["step"] == 5:
                    if not re.fullmatch(r'\d{3}', raw.strip()):
                        return "loan_eligibility_check", {}, "Enter a valid CIBIL score (300–900)."
                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 700:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: CIBIL score must be 700 or above."
                    L["step"] = 6
                    return "loan_eligibility_check", {}, "Enter your property market value (₹)."

                # Step 6: Property Value → Calculate Max Loan (LTV ≈ 50%)
                if L["step"] == 6:
                    amt = re.search(r'\d{5,12}', raw.replace(',', ''))
                    if not amt:
                        return "loan_eligibility_check", {}, "Please enter property value numeric. Example: 3500000"
                    prop_value = int(amt.group(0))

                    loan_amt = int(prop_value * 0.50)

                    E["eligible"] = True
                    E["amount"] = loan_amt
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    return (
                        "loan_eligibility_result",
                        {},
                        f"Loan Against Property Eligibility Confirmed.\n\n"
                        f"Property Value: ₹{prop_value:,}\n"
                        f"Eligible Loan Amount (50% LTV): ₹{loan_amt:,}\n"
                        f"Repayment Tenure: Up to 15 years\n\n"
                        f"Type 'apply' to continue or 'not now' to cancel."
                    )

            # ===== GOLD LOAN RULES =====
            if L["product"] == "gold":
                # Step 1: Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 18:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum age is 18."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "Enter gold weight in grams (e.g., 25)."

                # Step 2: Gold Weight
                if L["step"] == 2:
                    w = re.search(r'(\d{1,4})', raw.strip())
                    if not w:
                        return "loan_eligibility_check", {}, "Please enter a valid weight in grams."
                    E["weight"] = int(w.group(1))
                    if E["weight"] < 5:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum 5 grams required."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Enter gold purity (carat) e.g., 22 or 24."

                # Step 3: Purity
                if L["step"] == 3:
                    if raw.strip() not in ["22", "23", "24"]:
                        return "loan_eligibility_check", {}, "Gold must be at least 22 carat. Please enter 22, 23, or 24."
                    E["purity"] = int(raw.strip())

                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter today's approximate gold market price per gram (₹). Example: 5800"

                # Step 4: Calculate Loan
                if L["step"] == 4:
                    price = re.search(r'(\d{3,6})', raw.replace(',', ''))
                    if not price:
                        return "loan_eligibility_check", {}, "Enter numeric price per gram. Example: 5800"
                    price_per_gram = int(price.group(1))

                    # Calculate value and loan LTV
                    gold_value = E["weight"] * price_per_gram
                    loan_amt = int(gold_value * 0.75)  # 75% LTV

                    E["eligible"] = True
                    E["amount"] = loan_amt
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    return (
                        "loan_eligibility_result",
                        {},
                        f"Gold Loan Eligibility Confirmed.\n\n"
                        f"Gold Weight: {E['weight']} g\n"
                        f"Purity: {E['purity']} Carat\n"
                        f"Estimated Gold Value: ₹{gold_value:,}\n"
                        f"Eligible Loan Amount (75% LTV): ₹{loan_amt:,}\n"
                        f"Tenure: Up to 12 months\n\n"
                        f"Type 'apply' to continue or 'not now' to cancel."
                    )
            # ===== LOAN AGAINST FIXED DEPOSIT (FD LOAN) =====
            if L["product"] == "fd":
                # Step 1: Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 18:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum age is 18."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "Enter your Fixed Deposit amount (₹)."

                # Step 2: FD Amount
                if L["step"] == 2:
                    amt = re.search(r'(\d{4,9})', raw.replace(',', ''))
                    if not amt:
                        return "loan_eligibility_check", {}, "Please enter a valid deposit amount (e.g., 50000)."
                    fd_amt = int(amt.group(1))
                    if fd_amt < 10000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: FD amount must be at least ₹10,000."
                    
                    # Calculate eligible loan (90% of FD)
                    loan_amt = int(fd_amt * 0.90)

                    E["eligible"] = True
                    E["amount"] = loan_amt
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    return (
                        "loan_eligibility_result",
                        {},
                        f"Loan Against FD Eligibility Confirmed.\n\n"
                        f"FD Amount: ₹{fd_amt:,}\n"
                        f"Eligible Loan Amount (90% of FD): ₹{loan_amt:,}\n"
                        f"Tenure: Up to remaining FD maturity\n"
                        f"Interest Rate: FD rate + ~1%\n\n"
                        f"Type 'apply' to continue or 'not now' to cancel."
                    )
            
            # ===== CREDIT CARD (Revolving Loan) Eligibility (CBI Rules) =====
            if L["product"] == "credit":

                # Step 1 — Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter your age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 21:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum age for Credit Card is 21."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "Enter your monthly income (₹). Minimum ₹20,000 required."

                # Step 2 — Income
                if L["step"] == 2:
                    m = re.search(r'(\d{4,9})', raw.replace(",", ""))
                    if not m:
                        return "loan_eligibility_check", {}, "Please enter numeric income, e.g., 25000."
                    E["salary"] = int(m.group(1))
                    if E["salary"] < 20000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum monthly income is ₹20,000."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Employment type?\n1) Government\n2) Private\n3) Self-Employed"

                # Step 3 — Employment Type
                if L["step"] == 3:
                    if raw.strip() == "1":
                        E["emp"] = "govt"
                    elif raw.strip() == "2":
                        E["emp"] = "private"
                    elif raw.strip() == "3":
                        E["emp"] = "self"
                    else:
                        return "loan_eligibility_check", {}, "Please choose 1 for Government, 2 for Private, 3 for Self-Employed."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter your CIBIL score (300–900). Must be 750+."

                # Step 4 — CIBIL Score
                if L["step"] == 4:

                    if re.search(r'what.*cibil|cibil meaning|explain cibil', text):
                        return "loan_info", {}, (
                            "CIBIL score is your credit repayment history rating.\n"
                            "Range: 300–900. Score above 750 is considered good.\n\n"
                            "Please enter your CIBIL score (300–900)."
                        )

                    if not re.fullmatch(r'\d{3,4}', raw.strip()):
                        return "loan_eligibility_check", {}, "Enter a valid CIBIL score (300–900)."

                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 750:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: CIBIL must be 750+ for Credit Card."

                    # Calculate Credit Limit (2x to 4x salary)
                    limit = E["salary"] * 3

                    E["eligible"] = True
                    E["amount"] = limit
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    return "loan_eligibility_result", {}, (
                        f"Credit Card Eligibility Confirmed.\n\n"
                        f"Estimated Credit Limit: ₹{limit:,}\n"
                        f"CIBIL Score Verified: {E['cibil']}\n"
                        f"Interest on Revolving Balance: ~2.50% per month (approx. 34% annually)\n"
                        f"Minimum Due Each Month: 5% of outstanding billing amount\n\n"
                        f"Note: If full bill is not paid, interest applies from the transaction date.\n\n"
                        f"Type 'apply' to continue or 'not now' to cancel."
                    )

            # ===== DEBT CONSOLIDATION LOAN ELIGIBILITY (CBI Rules) =====
            if L["product"] == "debt":

                # Step 1 — Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Enter your age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 21:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum age is 21."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "Enter your monthly income (₹). Minimum ₹25,000 required."

                # Step 2 — Monthly Income
                if L["step"] == 2:
                    m = re.search(r'(\d{4,9})', raw.replace(",", ""))
                    if not m:
                        return "loan_eligibility_check", {}, "Please enter numeric income, e.g., 30000."
                    E["salary"] = int(m.group(1))
                    if E["salary"] < 25000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum monthly income is ₹25,000."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Employment Type?\n1) Government\n2) Private\n3) Self-Employed"

                # Step 3 — Employment Type
                if L["step"] == 3:
                    if raw.strip() == "1":
                        E["emp"] = "govt"
                    elif raw.strip() == "2":
                        E["emp"] = "private"
                    elif raw.strip() == "3":
                        E["emp"] = "self"
                    else:
                        return "loan_eligibility_check", {}, "Choose 1 for Government, 2 for Private, 3 for Self-Employed."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "How many years of work experience?"

                # Step 4 — Experience Check
                if L["step"] == 4:
                    if not re.fullmatch(r'\d{1,2}', raw.strip()):
                        return "loan_eligibility_check", {}, "Enter years as a number."
                    E["exp"] = int(raw.strip())

                    if E["emp"] == "govt" and E["exp"] < 1:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Government employees need at least 1 year experience."

                    if E["emp"] == "private" and E["exp"] < 3:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Private employees need at least 3 years experience."

                    if E["emp"] == "self" and E["exp"] < 2:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Self-employed need minimum 2 years experience + ITR."

                    L["step"] = 5
                    return "loan_eligibility_check", {}, "Enter your CIBIL score (300–900). Minimum 700 required."

                # Step 5 — CIBIL Score
                if L["step"] == 5:

                    if re.search(r'what.*cibil|explain cibil|cibil meaning', text):
                        return "loan_info", {}, (
                            "CIBIL score represents your credit repayment history.\n"
                            "Range: 300–900. Higher score = easier loan approvals.\n\n"
                            "Enter your CIBIL score now (300–900)."
                        )

                    if not re.fullmatch(r'\d{3,4}', raw.strip()):
                        return "loan_eligibility_check", {}, "Please enter a valid CIBIL score (300–900)."

                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 700:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: CIBIL score must be 700+ for Debt Consolidation."

                    # Calculate Eligible Loan
                    amount = E["salary"] * 20
                    E["eligible"] = True
                    E["amount"] = amount
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    return "loan_eligibility_result", {}, (
                        f"Debt Consolidation Loan Eligibility Confirmed.\n\n"
                        f"Maximum Eligible Loan Amount: ₹{amount:,}\n"
                        f"CIBIL Score Verified: {E['cibil']}\n"
                        f"Tenure: Up to 7 years\n\n"
                        f"This loan will combine your existing EMIs into one single payment.\n\n"
                        f"Type 'apply' to continue or 'not now' to cancel."
                    )


            # ===== TERM LOAN ELIGIBILITY =====
            if L["product"] == "term":
                E = L["elig"]

                # Step 1: Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 21:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum age is 21."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "How many years has your business been running?"

                # Step 2: Business Vintage
                if L["step"] == 2:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Enter years in numbers."
                    E["exp"] = int(raw.strip())
                    if E["exp"] < 2:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Business must run for at least 2 years."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Enter your annual business turnover (₹)."

                # Step 3: Turnover
                if L["step"] == 3:
                    m = re.search(r'\d{4,10}', raw.replace(',',''))
                    if not m:
                        return "loan_eligibility_check", {}, "Enter numeric amount, example: 350000"
                    E["turnover"] = int(m.group())
                    if E["turnover"] < 300000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum turnover should be ₹3,00,000/year."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter your CIBIL score (300–900)."

                # Step 4: CIBIL Check
                if L["step"] == 4:
                    if not re.fullmatch(r'\d{3}', raw.strip()):
                        return "loan_eligibility_check", {}, "Enter valid CIBIL score (300–900)."
                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 700:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: CIBIL must be 700 or above."
                    L["step"] = 5
                    return "loan_eligibility_check", {}, "Loan Purpose?\n1) Business Expansion\n2) Machinery / Equipment\n3) Office/Shop Renovation"

                # Step 5: Purpose + Final Output
                if L["step"] == 5:
                    E["purpose"] = raw.strip()
                    amount = max(150000, int(E["turnover"] * 0.4))  # approx formula
                    E["eligible"] = True
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    if E["cibil"] >= 800:
                        guarantor_msg = "Not required (Strong CIBIL)"
                    else:
                        guarantor_msg = "1 guarantor recommended"

                    return "loan_eligibility_result", {}, (
                        f"Term Loan Eligibility Confirmed.\n\n"
                        f"Estimated Eligible Loan Amount: ₹{amount:,}\n"
                        f"Guarantor Requirement: {guarantor_msg}\n"
                        f"Repayment Tenure: Up to 7 years\n\n"
                        "Type 'apply' to continue or 'not now' to cancel."
                    )
   
            # ===== WORKING CAPITAL LOAN (WC LOAN) =====
            if L["product"] == "wc":
                E = L["elig"]

                # Step 1: Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 21:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum age is 21."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "How many years has your business been running?"

                # Step 2: Business Vintage (Minimum 1 year)
                if L["step"] == 2:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Enter years as a number."
                    E["exp"] = int(raw.strip())
                    if E["exp"] < 1:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Business must run for at least 1 year."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Enter your annual business turnover (₹)."

                # Step 3: Turnover
                if L["step"] == 3:
                    m = re.search(r'\d{4,10}', raw.replace(',',''))
                    if not m:
                        return "loan_eligibility_check", {}, "Enter numeric amount, example: 350000"
                    E["turnover"] = int(m.group())
                    if E["turnover"] < 300000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum turnover should be ₹3,00,000/year."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter your Business Credit Score (CIBIL / Commercial Score)."

                # Step 4: Credit Score
                if L["step"] == 4:
                    if not re.fullmatch(r'\d{3}', raw.strip()):
                        return "loan_eligibility_check", {}, "Enter valid score (300–900)."
                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 600:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Business credit score must be at least 600."
                    L["step"] = 5
                    return "loan_eligibility_check", {}, "Business Type?\n1) Trading\n2) Manufacturing\n3) Services"

                # Step 5: Final Calculation
                if L["step"] == 5:
                    E["type"] = raw.strip()
                    amount = max(100000, int(E["turnover"] * 0.20))  # 20% of turnover

                    if amount > 2500000:
                        amount = 2500000

                    E["eligible"] = True
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    return "loan_eligibility_result", {}, (
                        f"Working Capital Loan Eligibility Confirmed.\n\n"
                        f"Eligible Loan Limit: ₹{amount:,}\n"
                        f"Repayment: Renewable every 12 months\n"
                        f"Security: May require property/business collateral\n\n"
                        "Type 'apply' to continue or 'not now' to cancel."
                    )

            # ===== EQUIPMENT FINANCING LOAN =====
            if L["product"] == "equip":
                E = L["elig"]

                # Step 1: Age
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter age in numbers."
                    E["age"] = int(raw.strip())
                    if E["age"] < 21:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum age is 21."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "How many years has your business been running?"

                # Step 2: Business Vintage (≥ 3 years)
                if L["step"] == 2:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Enter number of years."
                    E["exp"] = int(raw.strip())
                    if E["exp"] < 3:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Business must be running for at least 3 years."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Enter your annual business turnover (₹)."

                # Step 3: Turnover
                if L["step"] == 3:
                    m = re.search(r'\d{4,10}', raw.replace(',',''))
                    if not m:
                        return "loan_eligibility_check", {}, "Enter numeric turnover, example: 1200000"
                    E["turnover"] = int(m.group())
                    if E["turnover"] < 500000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum turnover required is ₹5,00,000 per year."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter your Credit Score (300–900)."

                # Step 4: Credit Score
                if L["step"] == 4:
                    if not re.fullmatch(r'\d{3}', raw.strip()):
                        return "loan_eligibility_check", {}, "Enter valid score (300–900)."
                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 750:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Credit Score must be 750 or above."
                    L["step"] = 5
                    return "loan_eligibility_check", {}, "Do you have GST & Udyam Registration? (yes/no)"

                # Step 5: Registration Confirmation
                if L["step"] == 5:
                    if "yes" in text:
                        # Calculate eligible loan
                        amount = int(E["turnover"] * 0.20)
                        if amount < 100000:
                            amount = 100000
                        if amount > 5000000:
                            amount = 5000000

                        E["eligible"] = True
                        L["service"] = "apply"
                        L["step"] = 10
                        memory["loan"]["waiting_apply"] = True

                        return "loan_eligibility_result", {}, (
                            f"Equipment Financing Loan Eligibility Confirmed.\n\n"
                            f"Eligible Loan Amount: ₹{amount:,}\n"
                            f"Repayment Tenure: Up to 7 years\n"
                            f"Margin Requirement: 25%\n"
                            f"Security: Hypothecation of Equipment + Business Collateral if required\n\n"
                            "Type 'apply' to continue or 'not now' to cancel."
                        )
                    else:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: GST & Udyam registration are mandatory."
                    
            # ===== INVOICE FINANCING (CENT TReDS) =====
            if L["product"] == "invoice":
                E = L["elig"]

                # Step 1: Business vintage
                if L["step"] == 1:
                    if not raw.strip().isdigit():
                        return "loan_eligibility_check", {}, "Please enter business age in years."
                    E["exp"] = int(raw.strip())
                    if E["exp"] < 2:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Business must be running for at least 2 years."
                    L["step"] = 2
                    return "loan_eligibility_check", {}, "Do you have Udyam (MSME) Registration? (yes/no)"

                # Step 2: Udyam check
                if L["step"] == 2:
                    if "yes" not in text:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: MSME (Udyam) registration is required."
                    L["step"] = 3
                    return "loan_eligibility_check", {}, "Enter your Credit Score (300–900)."

                # Step 3: CIBIL Score
                if L["step"] == 3:
                    if not re.fullmatch(r'\d{3}', raw.strip()):
                        return "loan_eligibility_check", {}, "Enter valid credit score."
                    E["cibil"] = int(raw.strip())
                    if E["cibil"] < 700:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Credit Score must be 700 or above."
                    L["step"] = 4
                    return "loan_eligibility_check", {}, "Enter invoice amount (₹)."

                # Step 4: Invoice amount
                if L["step"] == 4:
                    amt = re.search(r'\d{4,12}', raw.replace(',',''))
                    if not amt:
                        return "loan_eligibility_check", {}, "Enter numeric invoice amount. Example: 250000"
                    invoice = int(amt.group())
                    if invoice < 25000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Minimum invoice value is ₹25,000."
                    if invoice > 10000000:
                        reset_loan(); memory["menu"]=None
                        return "loan_eligibility_result", {}, "Not eligible: Maximum invoice allowed is ₹1,00,00,000 per invoice."

                    E["amount"] = invoice
                    finance = int(invoice * 0.80)

                    E["eligible"] = True
                    L["service"] = "apply"
                    L["step"] = 10
                    memory["loan"]["waiting_apply"] = True

                    return "loan_eligibility_result", {}, (
                        f"Invoice Financing Eligibility Confirmed.\n\n"
                        f"Invoice Value: ₹{invoice:,}\n"
                        f"Eligible Finance: ₹{finance:,}\n"
                        f"Repayment: 30–90 days (Buyer pays bank)\n"
                        f"Platform: CENT TReDS (RXIL)\n\n"
                        "Type 'apply' to continue or 'not now' to cancel."
                    )
        
            # ===== OVERDRAFT FACILITY (OD) =====
            if L["product"] == "od":
                E = L["elig"]

                # Step 1: Personal vs Business
                if L["step"] == 1:
                    if raw.strip() == "1":
                        E["type"] = "personal"
                        L["step"] = 2
                        return "loan_eligibility_check", {}, "Has your savings account been active for at least 6 months? (yes/no)"
                    elif raw.strip() == "2":
                        E["type"] = "business"
                        L["step"] = 5
                        return "loan_eligibility_check", {}, "How old is your business? (years)"
                    return "loan_eligibility_check", {}, "Please choose 1 for Personal or 2 for Business."

                # ----- PERSONAL OD -----
                if E.get("type") == "personal":
                    if L["step"] == 2:
                        if "yes" not in text:
                            reset_loan(); memory["menu"]=None
                            return "loan_eligibility_result", {}, "Not eligible: Account must be active for 6+ months."
                        L["step"] = 3
                        return "loan_eligibility_check", {}, "Is your Aadhaar linked to this account? (yes/no)"
                    if L["step"] == 3:
                        if "yes" not in text:
                            reset_loan(); memory["menu"]=None
                            return "loan_eligibility_result", {}, "Not eligible: Aadhaar linking is mandatory."
                        L["step"] = 4
                        return "loan_eligibility_check", {}, "Enter your average monthly balance (₹)."
                    if L["step"] == 4:
                        m = re.search(r'\d{3,9}', raw.replace(',',''))
                        if not m:
                            return "loan_eligibility_check", {}, "Enter numeric balance, e.g., 1500"
                        bal = int(m.group())
                        od_limit = min(bal * 4, 5000)
                        E["eligible"] = True
                        L["service"] = "apply"; L["step"] = 10; memory["loan"]["waiting_apply"] = True
                        return "loan_eligibility_result", {}, (
                            f"Overdraft Eligibility Confirmed.\n\n"
                            f"Eligible Limit: ₹{od_limit:,}\n"
                            f"Interest: Base Rate + 2%\n"
                            f"Tenure: Up to 36 months (with review)\n\n"
                            "Type 'apply' to continue or 'not now' to cancel."
                        )

                # ----- BUSINESS OD -----
                if E.get("type") == "business":
                    if L["step"] == 5:
                        if not raw.strip().isdigit() or int(raw.strip()) < 2:
                            reset_loan(); memory["menu"]=None
                            return "loan_eligibility_result", {}, "Not eligible: Business must be running for at least 2 years."
                        L["step"] = 6
                        return "loan_eligibility_check", {}, "Enter Credit Score (300–900)."
                    if L["step"] == 6:
                        if not re.fullmatch(r'\d{3}', raw.strip()):
                            return "loan_eligibility_check", {}, "Enter valid credit score."
                        score = int(raw.strip())
                        if score < 700:
                            reset_loan(); memory["menu"]=None
                            return "loan_eligibility_result", {}, "Not eligible: Credit score must be 700+."
                        L["step"] = 7
                        return "loan_eligibility_check", {}, "Enter annual turnover (₹)."
                    if L["step"] == 7:
                        amt = re.search(r'\d{4,12}', raw.replace(',',''))
                        if not amt:
                            return "loan_eligibility_check", {}, "Enter numeric turnover, e.g., 1200000"
                        turnover = int(amt.group())
                        od_limit = int(turnover * 0.20)
                        E["eligible"] = True
                        L["service"] = "apply"; L["step"] = 10; memory["loan"]["waiting_apply"] = True
                        return "loan_eligibility_result", {}, (
                            f"Business OD Eligibility Confirmed.\n\n"
                            f"Annual Turnover: ₹{turnover:,}\n"
                            f"Eligible OD Limit: ₹{od_limit:,} (20% of turnover)\n\n"
                            "Type 'apply' to continue or 'not now' to cancel."
                        )


            # ===== (Default) Non-education loans =====
            if L["step"] == 1:
                if not raw.strip().isdigit():
                    return "loan_eligibility_check", {}, "Please enter age in numbers."
                E["age"] = int(raw.strip())

                if E["age"] < 18 or E["age"] > 75:
                    reset_loan(); memory["menu"] = None
                    return "loan_eligibility_result", {}, "Not eligible: Age must be between 18 and 75."

                L["step"] = 2
                return "loan_eligibility_check", {}, "Enter your monthly income (₹): (Minimum ₹25,000 required)"

            if L["step"] == 2:
                m = re.search(r'(\d{4,9})', raw.replace(',', ''))
                if not m:
                    return "loan_eligibility_check", {}, "Please enter numeric income. Example: 30000"
                E["salary"] = int(m.group(1))

                if E["salary"] < 25000:
                    reset_loan(); memory["menu"] = None
                    return "loan_eligibility_result", {}, "Not eligible: Minimum income required is ₹25,000/month."

                L["step"] = 3
                return "loan_eligibility_check", {}, (
                    "Select Employment Type:\n"
                    "1) Government Employee\n"
                    "2) Private Employee\n"
                    "3) Self-Employed"
                )

            if L["step"] == 3:
                if raw.strip() == "1":
                    E["emp"] = "govt"
                elif raw.strip() == "2":
                    E["emp"] = "private"
                elif raw.strip() == "3":
                    E["emp"] = "self"
                else:
                    return "loan_eligibility_check", {}, (
                        "Please select:\n1) Government Employee\n2) Private Employee\n3) Self-Employed"
                    )
                L["step"] = 4
                return "loan_eligibility_check", {}, "Enter total years of work experience:"

            if L["step"] == 4:
                if not re.fullmatch(r'\d{1,2}', raw.strip()):
                    return "loan_eligibility_check", {}, "Please enter experience in years (1-30)."

                E["exp"] = int(raw.strip())
                
                # Employment-based experience rules
                if E["emp"] == "govt" and E["exp"] < 1:
                    reset_loan(); memory["menu"] = None
                    return "loan_eligibility_result", {}, "Not eligible: Government employees need minimum 1 year experience."
                if E["emp"] == "private" and E["exp"] < 3:
                    reset_loan(); memory["menu"] = None
                    return "loan_eligibility_result", {}, "Not eligible: Private employees need minimum 3 years experience."
                if E["emp"] == "self" and E["exp"] < 2:
                    reset_loan(); memory["menu"] = None
                    return "loan_eligibility_result", {}, (
                        "Not eligible: Self-employed individuals must have minimum **2 years business stability** "
                        "and valid ITR proof."
                    )

                L["step"] = 5
                return "loan_eligibility_check", {}, "Enter your CIBIL score (300–900):"

            if L["step"] == 5:
                if not re.fullmatch(r'\d{3,4}', raw.strip()):
                    return "loan_eligibility_check", {}, "Please enter valid CIBIL (300–900)."

                E["cibil"] = int(raw.strip())

                if E["cibil"] < 750:
                    reset_loan(); memory["menu"] = None
                    return "loan_eligibility_result", {}, "Not eligible: CIBIL score must be 750 or above."

                L["step"] = 6
                return "loan_eligibility_check", {}, "Enter property value (₹):"

            if L["step"] == 6:
                p = re.search(r'\d{5,10}', raw.replace(',', ''))
                if not p:
                    return "loan_eligibility_check", {}, "Please enter property value numeric. Example: 5000000"
                property_value = int(p.group(0))

                # LTV Calculation
                if property_value <= 3000000:
                    loan_amount = int(property_value * 0.90)
                elif property_value <= 7500000:
                    loan_amount = int(property_value * 0.80)
                else:
                    loan_amount = int(property_value * 0.75)

                E["eligible"] = True
                E["amount"] = loan_amount

                L["service"] = "apply"
                L["step"] = 10
                memory["loan"]["waiting_apply"] = True

                return "loan_eligibility_result", {}, (
                    f"Home Loan Eligibility Confirmed.\n\n"
                    f"Maximum Eligible Loan Amount: ₹{loan_amount:,}\n"
                    f"Co-Borrower Recommended: Yes (Parent/Spouse)\n"
                    f"Repayment Tenure: Up to 30 years (subject to retirement age)\n\n"
                    f"Type 'apply' to continue with application or 'not now' to exit."
                )

        # Apply (after eligibility OK)
                # Apply (after eligibility OK)
        # ================= APPLY SECTION (Separated for Secured / Unsecured / Business) =================
        if L["service"] == "apply":
            A = memory["loan"]["apply"]

            # --- Step 10: Full Name ---
            if L["step"] == 10:
                if re.fullmatch(r'[A-Za-z][A-Za-z .\-]{1,50}', raw):
                    A["name"] = raw.strip().title()

                    # Secured Loan (Home/Auto/LAP/Gold/FD)
                    if L["category"] == "secured":
                        L["step"] = 11
                        return "loan_apply_step", {}, "Please confirm your monthly salary (₹)."

                    # Unsecured Loan (Personal/Edu/Credit/Debt)
                    if L["category"] == "unsecured":
                        L["step"] = 11
                        return "loan_apply_step", {}, "Please enter your PAN Number (or type 'skip')."

                    # Business Loan
                    if L["category"] == "business":
                        L["step"] = 11
                        return "loan_apply_step", {}, "Enter your Business / Enterprise Name:"

                return "loan_apply_start", {}, "Please provide your full name."

            # ================= SECURED LOAN APPLY =================
            if L["category"] == "secured":

                # Step 11 — Salary
                if L["step"] == 11:
                    m = re.search(r'(\d{4,9})', raw.replace(',', ''))
                    if not m:
                        return "loan_apply_step", {}, "Please enter salary in numbers."
                    A["salary"] = int(m.group(1))
                    L["step"] = 12
                    return "loan_apply_step", {}, "Please enter your PAN (or type 'skip')."

                # Step 12 — PAN
                if L["step"] == 12:
                    if re.fullmatch(r'[A-Z]{5}\d{4}[A-Z]', raw.strip(), re.I):
                        A["pan"] = raw.strip().upper()
                    elif normalize_text(raw) == "skip":
                        A["pan"] = "Not Provided"
                    else:
                        return "loan_apply_step", {}, "Enter valid PAN (ABCDE1234F) or type 'skip'."

                    L["step"] = 13
                    return "loan_apply_upload", {}, (
                        "Please upload documents on Caashmora Official Portal.\n"
                        "If you need help, type 'what documents'. Once uploaded, type 'done'."
                    )

                # Step 13 — Upload + WHAT DOCUMENTS
                if L["step"] == 13:

                    if re.search(r'what.*document|document[s]?\??', text):
                        return "loan_required_documents", {}, DOCS_SECURED

                    if re.search(r'\b(done|submitted|uploaded)\b', text):
                        name = A["name"]; sal = A["salary"]; pan = A.get("pan", "Not Provided")
                        reset_loan(); memory["menu"] = None
                        return ("loan_apply_submit", {},
                                f"Secured Loan Application Submitted.\n"
                                f"Name: {name}\nSalary: ₹{sal:,}\nPAN: {pan}\n"
                                "Verification will begin shortly. You will get further updates via registered SMS/Email.\n\nWould you like to continue?")
                    
                    return "loan_apply_wait", {}, (
                        "Please upload documents on Caashmora Official Portal.\n"
                        "If you need help, type 'what documents'. Once uploaded, type 'done'."
                    )

            # ================= UNSECURED LOAN APPLY =================
            if L["category"] == "unsecured":

                # Step 11 — PAN
                if L["step"] == 11:
                    if re.fullmatch(r'[A-Z]{5}\d{4}[A-Z]', raw.strip(), re.I):
                        A["pan"] = raw.strip().upper()
                    elif normalize_text(raw) == "skip":
                        A["pan"] = "Not Provided"
                    else:
                        return "loan_apply_step", {}, "Enter valid PAN (ABCDE1234F) or type 'skip'."

                    L["step"] = 12
                    return "loan_apply_upload", {}, (
                        "Please upload documents on Caashmora Official Portal.\n"
                        "If you need help, type 'what documents'. Once uploaded, type 'done'."
                    )

                # Step 12 — Upload + WHAT DOCUMENTS
                if L["step"] == 12:

                    if re.search(r'what.*document|document[s]?\??', text):
                        return "loan_required_documents", {}, DOCS_UNSECURED

                    if re.search(r'\b(done|submitted|uploaded)\b', text):
                        name = A["name"]; pan = A.get("pan", "Not Provided")
                        reset_loan(); memory["menu"] = None
                        return ("loan_apply_submit", {},
                                f" Unsecured Loan Application Submitted.\n"
                                f"Name: {name}\nPAN: {pan}\n"
                                "Verification will begin shortly. You will get further updates via registered SMS/Email.\n\nWould you like to continue?")
                    
                    return "loan_apply_wait", {}, (
                        "Please upload documents on Caashmora Official Portal.\n"
                        "If you need help, type 'what documents'. Once uploaded, type 'done'."
                    )

            # ================= BUSINESS LOAN APPLY =================
            if L["category"] == "business":

                # Step 11 — Business Name
                if L["step"] == 11:
                    if re.fullmatch(r'[A-Za-z0-9 &.\-]{2,60}', raw):
                        A["business_name"] = raw.strip().title()
                        L["step"] = 12
                        return "loan_apply_step", {}, "Enter your GST Number (e.g., 33AAAAA1234A1Z5):"
                    return "loan_apply_step", {}, "Enter valid Business / Enterprise Name."

                # Step 12 — GST Number
                if L["step"] == 12:
                    if re.fullmatch(r'\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z\d]\d', raw.strip(), re.I):
                        A["gst"] = raw.strip().upper()
                        L["step"] = 13
                        return "loan_apply_upload", {}, (
                            "Please upload documents on Caashmora Official Portal.\n"
                            "If you need help, type 'what documents'. Once uploaded, type 'done'."
                        )
                    return "loan_apply_step", {}, "Invalid GST. Example: 33AAAAA1234A1Z5"

                # Step 13 — Upload + WHAT DOCUMENTS
                if L["step"] == 13:

                    if re.search(r'what.*document|document[s]?\??', text):
                        return "loan_required_documents", {}, DOCS_BUSINESS

                    if re.search(r'\b(done|submitted|uploaded)\b', text):
                        name = A["name"]; biz = A["business_name"]; gst = A["gst"]
                        reset_loan(); memory["menu"] = None
                        return ("loan_apply_submit", {},
                                f"Business Loan Application Submitted.\n"
                                f"Applicant: {name}\nBusiness: {biz}\nGST: {gst}\n"
                                "Verification will begin shortly. You will get further updates via registered SMS/Email.\n\nWould you like to continue?")
                    
                    return "loan_apply_wait", {}, (
                        "Please upload documents on Caashmora Official Portal.\n"
                        "If you need help, type 'what documents'. Once uploaded, type 'done'."
                    )



    # ===== OPEN ACCOUNT
    if re.search(r'\b(open account|create account|new account|open an account)\b', text) or text=="create account":
        memory["menu"]="account"; reset_acct()
        memory["acct"]["step"]=1
        return "account_open_start", {}, "Sure. Please provide your full name."

    if memory.get("menu")=="account":
        A = memory["acct"]
        if A["step"]==1:
            if not re.fullmatch(r'[A-Za-z][A-Za-z .\-]{1,50}', raw):
                return "account_open_start", {}, "Please enter a valid full name."
            A["name"]=raw.strip().title(); A["step"]=2
            return "account_open_step", {}, "Please enter your age in years."
        if A["step"]==2:
            if not raw.strip().isdigit():
                return "account_open_step", {}, "Please enter a valid age in numbers."
            age=int(raw.strip())
            if age<18:
                memory["menu"]=None; reset_acct()
                return "account_open_step", {}, "Minimum age is 18."
            A["age"]=age; A["step"]=3
            return "account_open_step", {}, "Please choose account type:\n1) Savings Account\n2) Current Account"
        if A["step"]==3:
            if raw.strip()=="1" or "savings" in text:
                A["type"]="Savings Account"
            elif raw.strip()=="2" or "current" in text:
                A["type"]="Current Account"
            else:
                return "account_open_step", {}, "Please choose 1 for Savings or 2 for Current."
            A["step"]=4
            return "account_open_step", {}, "Please enter your full address."
        if A["step"]==4:
            if len(raw)<8:
                return "account_open_step", {}, "Please enter a complete address."
            A["addr"]=raw.strip(); A["step"]=5
            return "account_open_step", {}, "Please enter your 12-digit Aadhaar number."
        if A["step"]==5:
            aad = re.sub(r'\D','',raw)
            if not re.fullmatch(r'\d{12}', aad):
                return "account_open_step", {}, "Please enter a valid 12-digit Aadhaar number."
            A["aadhaar"]=aad; A["step"]=6
            masked = mask_aadhaar(aad)
            return ("account_open_step", {},
                    f"Confirm details:\nName: {A['name']}\nAge: {A['age']}\nAccount Type: {A['type']}\n"
                    f"Address: {A['addr']}\nAadhaar: {masked}\nType 'confirm' to submit or 'edit' to restart.")
        if A["step"]==6:
            if normalize_text(raw)=="confirm":
                reset_acct(); memory["menu"]=None
                return "account_open_submit", {}, "Account opening request submitted. You will receive further updates via SMS/Email.\n\nWould you like to continue?"
            if normalize_text(raw)=="edit":
                reset_acct(); memory["menu"]="account"; memory["acct"]["step"]=1
                return "account_open_start", {}, "Okay, let's restart. Please provide your full name."
            return "account_open_step", {}, "Please type 'confirm' to submit or 'edit' to restart."

    # ===== EMI quick calc
    if re.search(r'\b(emi|emi calculator)\b', text):
        return "emi_calculator", {}, ("EMI Calculator:\nChoose type: monthly/quarterly/yearly\n"
                                      "Or enter like: '400000 48 months'.")

    if re.search(r'\bmonthly|quarterly|yearly\b', text) or re.search(r'\d+\s+(months?|years?|quarters?)', text):
        nums = re.findall(r'(\d{1,12})', raw)
        unit_m = bool(re.search(r'month', text))
        unit_y = bool(re.search(r'year', text))
        unit_q = bool(re.search(r'quarter', text))
        if len(nums)>=2:
            P = float(nums[0]); t = int(nums[1])
            if unit_y:
                n = t*12; r = ANNUAL_RATE/12
            elif unit_q:
                n = t*3; r = ANNUAL_RATE/12  # simplified
            else:
                n = t; r = ANNUAL_RATE/12
            try:
                emi_val = calculate_emi(P, r, n)
                return "emi_result", {}, f"Estimated EMI for ₹{int(P):,} over {n} months at {ANNUAL_RATE*100}% p.a. is ₹{int(round(emi_val))}/month."
            except Exception:
                pass

    # ===== generic info
    if re.search(r'cibil', text):
        return "loan_info", {}, "CIBIL score is a credit score from 300–900 that reflects your repayment history. Banks usually require 750+."
    if re.search(r'what document|which document|required document|document list', text):
        cat = memory["loan"]["category"]

        if cat == "secured":
            return "loan_required_documents", {}, DOCS_SECURED

        if cat == "unsecured":
            return "loan_required_documents", {}, DOCS_UNSECURED

        if cat == "business":
            return "loan_required_documents", {}, DOCS_BUSINESS

        # If no category yet (user asked before choosing loan)
        return "loan_required_documents", {}, (
            "Documents vary by loan type. Please select a loan category first."
        )

                
    # ===== dataset fallback if confident (safe; doesn't touch card actions)
    if has_data and model is not None:
        try:
            probs = model.predict_proba([raw])[0]
            pred = model.classes_[probs.argmax()]
            conf = probs.max()
        except Exception:
            pred, conf = "unknown", 0.0
        if conf >= CONFIDENCE_THRESHOLD:
            resp = dataset_response_for_intent(pred, raw)
            if resp:
                return pred, ent, resp
    
        # ===== SMALL TALK (THANK YOU / OK / BYE / NO) =====
    if text in ["thank you", "thankyou", "thanks", "thx"]:
        return "thanks", {}, "You're welcome! Let me know if you need anything else."

    if text in ["bye", "goodbye", "see you"]:
        return "goodbye", {}, "Thank you for using CAASHMORA Bank Assistant. Have a great day!"

    if text in ["ok", "okay", "k", "sure", "fine", "alright", "yes"]:
        return "acknowledge", {}, "Okay."

    if text in ["no", "not now", "later"]:
        return "reject", {}, "No problem, I'm here whenever you're ready."

    return "unknown", ent, "Sorry, I did not understand that. Could you please rephrase?"

# ========= CLI =========
def main():
    print("Welcome to CAASHMORA Bank Virtual Assistant (Milestone 2).\n")
    print("Type 'exit' to end the chat.\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not msg:
            continue

        if msg.lower() in ['exit', 'quit', 'bye']:
            print("\n🎯 Predicted Intent: exit")
            print("📎 Extracted Entities: {}")
            print("🤖 Bot: Thank you for using CAASHMORA Bank. Goodbye!")
            break

        intent, entities, reply = handle_user_input(msg)
        print(f"\n🎯 Predicted Intent: {intent}")
        print(f"📎 Extracted Entities: {entities if entities else {}}")
        print(f"🤖 Bot: {reply}\n")

if __name__ == "__main__":
    main()