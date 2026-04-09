from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os
import numpy as np
app = Flask(__name__)
app.secret_key = "fraud_project_secret"

DB_FILE = "transactions.db"
MODEL_FILE = "model_data.npz"

transaction_type_map = {
    "Transfer": 0,
    "Cash Out": 1,
    "Payment": 2,
    "Debit": 3
}

location_risk_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

device_type_map = {
    "Mobile": 0,
    "Laptop": 1,
    "ATM": 2
}

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def load_model():
    if not os.path.exists(MODEL_FILE):
        return None, None, None
    data = np.load(MODEL_FILE)
    return data["weights"], data["means"], data["stds"]

def predict_transaction(amount, transaction_type, old_balance, new_balance, location_risk, device_type, transaction_frequency):
    weights, means, stds = load_model()

    if weights is None:
        return None, None

    x = np.array([[
        amount,
        transaction_type_map[transaction_type],
        old_balance,
        new_balance,
        location_risk_map[location_risk],
        device_type_map[device_type],
        transaction_frequency
    ]], dtype=float)

    x_scaled = (x - means) / stds
    x_final = np.c_[np.ones(x_scaled.shape[0]), x_scaled]

    probability = float(sigmoid(np.dot(x_final, weights))[0])
    prediction = 1 if probability >= 0.5 else 0

    return prediction, probability

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL,
            transaction_type TEXT,
            old_balance REAL,
            new_balance REAL,
            location_risk TEXT,
            device_type TEXT,
            transaction_frequency INTEGER,
            result TEXT,
            probability REAL
        )
    """)
    conn.commit()
    conn.close()

def get_dashboard_stats():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM transactions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions WHERE result = 'Fraud'")
    fraud = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions WHERE result = 'Not Fraud'")
    safe = cursor.fetchone()[0]

    conn.close()

    fraud_rate = round((fraud / total) * 100, 2) if total > 0 else 0

    return {
        "total": total,
        "fraud": fraud,
        "safe": safe,
        "fraud_rate": fraud_rate
    }

init_db()

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "1234":
            session["user"] = username
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")

@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    stats = get_dashboard_stats()
    return render_template("index.html", stats=stats)

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        amount = float(request.form["amount"])
        transaction_type = request.form["transaction_type"]
        old_balance = float(request.form["old_balance"])
        new_balance = float(request.form["new_balance"])
        location_risk = request.form["location_risk"]
        device_type = request.form["device_type"]
        transaction_frequency = int(request.form["transaction_frequency"])

        prediction, probability = predict_transaction(
            amount, transaction_type, old_balance, new_balance,
            location_risk, device_type, transaction_frequency
        )

        if prediction is None:
            return "Model file not found. Run train_model.py first."

        result = "Fraud" if prediction == 1 else "Not Fraud"
        confidence = round(probability * 100, 2)

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transactions
            (amount, transaction_type, old_balance, new_balance, location_risk, device_type, transaction_frequency, result, probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            amount, transaction_type, old_balance, new_balance,
            location_risk, device_type, transaction_frequency,
            result, confidence
        ))
        conn.commit()
        conn.close()

        return render_template("result.html", result=result, confidence=confidence)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    search = request.args.get("search", "").strip()

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if search:
        query = """
            SELECT * FROM transactions
            WHERE
                CAST(id AS TEXT) LIKE ?
                OR CAST(amount AS TEXT) LIKE ?
                OR transaction_type LIKE ?
                OR location_risk LIKE ?
                OR device_type LIKE ?
                OR result LIKE ?
            ORDER BY id DESC
        """
        like_value = f"%{search}%"
        cursor.execute(query, (like_value, like_value, like_value, like_value, like_value, like_value))
    else:
        cursor.execute("SELECT * FROM transactions ORDER BY id DESC")

    data = cursor.fetchall()
    conn.close()

    return render_template("history.html", data=data, search=search)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
