from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import time

app = Flask(__name__)

# Carico il modello salvato
model = load('modello_regressione.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    time_start_latency = time.time()

    try:
        valore_esperienza = float(data['hyper_param'])
        input_df = pd.DataFrame([[valore_esperienza]], columns=["Years of Experience"])
        predizione = model.predict(input_df)[0]

        time_send_latency = time.time()
        latency = time_send_latency - time_start_latency

        return jsonify({
            'stipendio_previsto': predizione,
            'latency_seconds': latency
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
