import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

app = Flask(__name__)

# Load the trained machine learning model
loaded_model = pickle.load(open("model.pkl", "rb"))

# Initialize encoders
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()

# Preprocessing function to encode categorical variables
def encode_transaction_method(method):
    # Assuming label encoding is sufficient for 'Transaction Method'
    encoded_method = label_encoder.fit_transform([method])[0]
    return encoded_method

def encode_location(location):
    # Assuming label encoding is sufficient for 'Location'
    encoded_location = label_encoder.fit_transform([location])[0]
    return encoded_location

def encode_card_type(card_type):
    # Assuming one-hot encoding is needed for 'Type of Card'
    global one_hot_encoder
    encoded_card_type = one_hot_encoder.fit_transform([[card_type]])
    return encoded_card_type

def encode_bank(bank):
    # Assuming one-hot encoding is needed for 'Bank'
    global one_hot_encoder
    encoded_bank = one_hot_encoder.fit_transform([[bank]])
    return encoded_bank

# Prediction function
def ValuePredictor(to_predict_list):
    # Preprocess the input data
    time, amount, transaction_method, transaction_id, card_type, location, bank = to_predict_list
    
    # Convert necessary input values to numerical format
    time = float(time)
    amount = float(amount)
    transaction_id = float(transaction_id)
    
    # Encode categorical variables
    transaction_method_encoded = encode_transaction_method(transaction_method)
    location_encoded = encode_location(location)
    card_type_encoded = encode_card_type(card_type)
    bank_encoded = encode_bank(bank)
    
    # Make prediction using preprocessed input
    to_predict = np.array([time, amount, transaction_id, transaction_method_encoded, location_encoded])
    to_predict = np.concatenate([to_predict, card_type_encoded.toarray()[0], bank_encoded.toarray()[0]]).reshape(1, -1)
    
    result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        time = request.form['time']
        amount = request.form['amount']
        transaction_method = request.form['tm']
        transaction_id = request.form['ti']
        card_type = request.form['ct']
        location = request.form['location']
        bank = request.form['em']
        
        # Make prediction
        result = ValuePredictor([time, amount, transaction_method, transaction_id, card_type, location, bank])
        
        # Process prediction result
        if int(result) == 1:
            prediction = 'Given transaction is fraudulent'
        else:
            prediction = 'Given transaction is NOT fraudulent'
            
        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
