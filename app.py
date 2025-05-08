from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import os
import csv

app = Flask(__name__)
app.secret_key = 'your_secret_key'

CSV_FILE_PATH = os.path.join(os.getcwd(), 'college_details_extended.csv')
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Dummy training data for Naive Bayes
X = np.array([
    [1, 180],
    [0, 150],
    [1, 160],
    [0, 190],
    [1, 140]
])
y = np.array([1, 0, 0, 1, 0])  # Labels

model = GaussianNB()
model.fit(X, y)

registered_users = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username in registered_users and registered_users[username] == password:
        session['username'] = username
        return redirect(url_for('dashboard'))
    return "Login Failed. Check username or password."

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    registered_users[username] = password
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('home'))

    colleges_data = []
    try:
        with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                colleges_data.append(row)
    except FileNotFoundError:
        colleges_data = None

    return render_template('home.html', username=session['username'], colleges=colleges_data)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'username' not in session:
        return redirect(url_for('home'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        filepath = os.path.join(os.getcwd(), 'college_details_extended.csv')
        file.save(filepath)
        return redirect(url_for('dashboard'))
    return "Invalid file format. Please upload a CSV file."

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('home'))

    # Get and store all inputs
    search_criteria = {
        'government_type': request.form.get('government_type', '').strip(),
        'hostel_status': request.form.get('hostel_status', '').strip(),
        'cutoff': request.form.get('cutoff', '').strip(),
        'fees': request.form.get('fees', '').strip()
    }

    # Convert to lowercase for processing
    government_type = search_criteria['government_type'].lower()
    hostel_status = search_criteria['hostel_status'].lower()
    cutoff = search_criteria['cutoff']
    fees = search_criteria['fees']

    # Load data
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Preprocess data
    df['Gov_Type_Lower'] = df['Government Type'].str.lower().str.strip()
    df['Hostel_Stats_Lower'] = df['Hostel Stats'].str.lower().str.strip()
    df['Fees'] = pd.to_numeric(df['Fees'], errors='coerce')
    df['Cutoff_Numeric'] = df['Cutoff'].str.extract(r'(\d+)').astype(float)
    
    # Initialize match score
    df['Match_Score'] = 0
    
    # Apply exact matching first
    if government_type:
        df['Match_Score'] += (df['Gov_Type_Lower'] == government_type) * 2
    
    if hostel_status:
        if 'girls' in hostel_status:
            df['Match_Score'] += df['Hostel_Stats_Lower'].str.contains('girls') * 2
        elif 'boys' in hostel_status:
            df['Match_Score'] += df['Hostel_Stats_Lower'].str.contains('boys') * 2
        elif 'both' in hostel_status:
            df['Match_Score'] += df['Hostel_Stats_Lower'].str.contains('both') * 2
        else:
            df['Match_Score'] += (df['Hostel_Stats_Lower'] == hostel_status) * 2
    
    if fees:
        try:
            fees_val = float(fees)
            df['Match_Score'] += (df['Fees'] == fees_val) * 3
        except ValueError:
            pass
    
    # Apply cutoff range filter (±10)
    if cutoff:
        try:
            cutoff_val = float(cutoff)
            df = df[
                (df['Cutoff_Numeric'] >= cutoff_val - 10) & 
                (df['Cutoff_Numeric'] <= cutoff_val + 10)
            ]
            df['Match_Score'] += (10 - abs(df['Cutoff_Numeric'] - cutoff_val)) * 0.5
        except ValueError:
            pass
    
    # Sort results
    df = df.sort_values(by=['Match_Score', 'Cutoff_Numeric'], ascending=[False, True])
    
    # Prediction logic
    prediction_result = None
    if government_type and cutoff and hostel_status:
        try:
            gov_map = {"government": 1, "private": 0, "government-aided": 2}
            hostel_map = {
                "available for girls": 1,
                "available for boys": 2, 
                "available for both": 3,
                "not available": 0,
                "limited": 0
            }
            
            gov_val = gov_map.get(government_type, 1)
            hostel_val = hostel_map.get(hostel_status, 0)
            cutoff_num = float(cutoff)
            
            input_data = np.array([[gov_val, cutoff_num, hostel_val]])
            
            # Model training data (including hostel status)
            X = np.array([
                [1, 180, 3], [0, 150, 0], [1, 160, 1],
                [0, 190, 2], [1, 140, 0]
            ])
            y = np.array([1, 0, 0, 1, 0])
            
            model = GaussianNB()
            model.fit(X, y)
            
            pred_label = model.predict(input_data)[0]
            prediction_result = "Above threshold" if pred_label == 1 else "Below threshold"
        except ValueError:
            prediction_result = "Invalid input values"

    # Prepare output
    columns_to_drop = [
        'Gov_Type_Lower', 'Hostel_Stats_Lower', 
        'Cutoff_Numeric', 'Match_Score',
        'Hostel for Boys', 'Hostel for Girls'
    ]
    
    return render_template(
        'home.html',
        prediction=prediction_result,
        colleges=df.drop(columns_to_drop, axis=1).to_dict('records'),
        username=session['username'],
        search_criteria=search_criteria  # Pass the original inputs back
    )
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
