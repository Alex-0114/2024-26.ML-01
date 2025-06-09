from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Carico il modello salvato
model = load('modello_regressione.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        valore_esperienza = float(data['hyper_param'])
        input_df = pd.DataFrame([[valore_esperienza]], columns=["Years of Experience"])
        predizione = model.predict(input_df)[0]

        return jsonify({'stipendio_previsto': predizione})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
