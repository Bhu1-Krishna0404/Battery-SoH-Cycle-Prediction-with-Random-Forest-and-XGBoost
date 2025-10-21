# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Dummy credentials for login
USERNAME = 'admin'
PASSWORD = 'admin'

# Load models and utilities
soh_model = joblib.load("soh_model.pkl")
rul_model = joblib.load("rul_model.pkl")
scaler = joblib.load("scaler.pkl")
le_battery = joblib.load("battery_le.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == USERNAME and password == PASSWORD:
            session['user'] = username
            return redirect(url_for('predict'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template('login.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # For now, save to session (or you could use a file/db)
        session['registered_user'] = username
        session['registered_pass'] = password

        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')



@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        flash('Please log in to continue.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            battery_id = request.form['battery_id']
            cycle = float(request.form['cycle'])
            chI = float(request.form['chI'])
            chV = float(request.form['chV'])
            chT = float(request.form['chT'])
            disI = float(request.form['disI'])
            disV = float(request.form['disV'])
            disT = float(request.form['disT'])
            BCt = float(request.form['BCt'])

            battery_encoded = le_battery.transform([battery_id])[0]
            input_features = np.array([[battery_encoded, cycle, chI, chV, chT, disI, disV, disT, BCt]])
            input_scaled = scaler.transform(input_features)

            predicted_soh = soh_model.predict(input_scaled)[0]
            predicted_rul = rul_model.predict(input_scaled)[0]

            return render_template('result.html',
                                   soh=round(predicted_soh, 2),
                                   rul=int(predicted_rul),
                                   battery_id=battery_id,
                                   cycle=int(cycle), chI=chI, chV=chV, chT=chT,
                                   disI=disI, disV=disV, disT=disT, BCt=BCt)

        except Exception as e:
            return f"Prediction failed: {e}"

    return render_template('predict.html')

@app.route('/performance')
def chart():
    return render_template('perfromance.html')


if __name__ == '__main__':
    app.run(debug=True)
