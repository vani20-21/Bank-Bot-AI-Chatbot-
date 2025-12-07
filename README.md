## BANK BOT AI – Smart Banking Virtual Assistant for CAASHMORA Bank

*Project completed under Infosys Springboard 6.0*

### Project Overview

Bank Bot AI is a virtual banking assistant that enables customers to access basic banking functionalities such as balance enquiry, fund transfer simulation, card services, loan information, and general banking queries.
It uses Machine Learning for intent classification and a secure backend built using Flask and SQLite.
An integrated Admin Dashboard allows monitoring of chatbot performance, managing FAQs, and retraining the model.

---

### Objectives

* Provide an interactive conversational banking interface
* Automate customer support operations
* Enable secure account and user management
* Provide real-time analytics to administrators
* Continuously improve chatbot intelligence with retraining

---

### Key Features

| Category          | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| User Portal       | Login, dashboard view, chatbot assistance, view chat logs            |
| Chatbot Engine    | Trained ML model to classify 72+ intents with high accuracy          |
| Banking Support   | Balance enquiry, transaction history, card assistance, loan process  |
| Admin Dashboard   | Shows analytics: Total queries, total intents, accuracy, recent logs |
| Data Management   | Export chat logs, add/delete FAQs                                    |
| Model Improvement | Retrain ML model via admin panel                                     |

---

### Technology Stack

| Component        | Technology Used                                 |
| ---------------- | ----------------------------------------------- |
| Frontend         | HTML, CSS, JavaScript                           |
| Backend          | Python Flask Framework                          |
| Database         | SQLite                                          |
| Machine Learning | Scikit-learn, TF-IDF Vectorizer, SVM Classifier |
| Version Control  | Git & GitHub                                    |

---

### Dataset and Model Details

| Parameter              | Value                       |
| ---------------------- | --------------------------- |
| Total training samples | 30,005                      |
| Total intents          | 72                          |
| Model accuracy         | 99.3% (Training evaluation) |
| Training dataset       | bankbot_final_expanded1.csv |

---

### Folder Structure

```
Bank-Bot-AI-Chatbot
 ├─ static/                   → CSS, images, JS files
 ├─ templates/                → HTML pages for UI
 ├─ app.py                    → Main application script
 ├─ milestone_one.py          → Rule-based intent processing
 ├─ milestone_two.py          → ML-based chatbot logic
 ├─ db.py                     → Database query functions
 ├─ setup_admin.py            → Creates admin account
 ├─ setup_users.py            → Inserts sample user accounts
 ├─ bank.db                   → Local SQLite database
 ├─ bankbot_final_expanded1.csv → Dataset used for model training
 └─ README.md                 → Documentation
```

---

### Steps to Run the Application

1. Install dependencies

```
pip install -r requirements.txt
```

2. Initialize admin and test user accounts

```
python setup_admin.py
python setup_users.py
```

3. Start the Flask server

```
python app.py
```

4. Open in browser

```
http://127.0.0.1:5000/
```

---

### Sample Login Credentials

#### Admin Login

| Email                                                 | Password  |
| ----------------------------------------------------- | --------- |
| [admin@caashmora.ac.in](mailto:admin@caashmora.ac.in) | admin@123 |

#### User Logins (CAASHMORA Bank Accounts)

| Account Number | Password      | Name        | Email                                               | Phone      | Starting Balance |
| -------------- | ------------- | ----------- | --------------------------------------------------- | ---------- | ---------------- |
| 8123623741     | Muruga@123    | Muruga S    | [muruga.ca@gmail.com](mailto:muruga.ca@gmail.com)   | 6513429873 | ₹200000          |
| 8912672463     | Tharunika@123 | Tharunika S | [tharunika3@gmail.com](mailto:tharunika3@gmail.com) | 9812327638 | ₹420000          |
| 23647126543    | Krishna@123   | Krishna P   | [Krishnna4@gmail.com](mailto:Krishnna4@gmail.com)   | 9856437865 | ₹300000          |

---

### Future Enhancements

* Multi-language chatbot support
* Live UPI/Banking API integration
* Mobile app version
* WhatsApp/Telegram bot deployment
* Voice-based interactions

---

### Declaration

This project is developed for academic and learning purposes as part of the **Infosys Springboard 6.0** program.
No real banking transactions are conducted.
