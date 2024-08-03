from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    data_df = pd.DataFrame([data])
    
    data_df['Chol_to_BP_Ratio'] = data_df['serumcholestrol'] / data_df['restingBP']
    data_df['MaxHR_to_Age'] = data_df['maxheartrate'] / data_df['age']
    
    data_scaled = scaler.transform(data_df)
    
    prediction = model.predict(data_scaled)[0]
    
    prediction_text = 'Positive for Cardiovascular Disease' if prediction == 1 else 'Negative for Cardiovascular Disease'
    
    return jsonify({'prediction': prediction_text})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
