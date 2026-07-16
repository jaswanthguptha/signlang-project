import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'healthy', 'message': 'ASL Sign Language API is running'}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/version', methods=['GET'])
def version():
    return jsonify({
        'status': 'success',
        'backend_version': '1.0.0-mock',
        'model_version': 'mocked',
        'label_map_version': 'mocked'
    }), 200

@app.route('/predict_alpha', methods=['POST'])
@app.route('/predict_alphabet', methods=['POST'])
def predict_alphabet():
    return jsonify({
        'status': 'success',
        'prediction': 'B',
        'confidence': 0.95,
        'predictions': [{'class': 'B', 'confidence': 0.95}]
    }), 200

@app.route('/predict_from_landmarks', methods=['POST'])
def predict_from_landmarks():
    return jsonify({
        'status': 'success',
        'prediction': 'B',
        'confidence': 0.95
    }), 200

@app.route('/predict_word', methods=['POST'])
def predict_word():
    return jsonify({
        'status': 'success',
        'prediction': 'hello',
        'confidence': 0.92
    }), 200

@app.route('/predict_debug', methods=['POST'])
def predict_debug():
    return jsonify({'status': 'success'}), 200

@app.route('/save_live_test', methods=['POST'])
def save_live_test():
    return jsonify({'status': 'success'}), 200

@app.route('/reset_sequence', methods=['POST'])
def reset_sequence():
    return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  Server starting on http://0.0.0.0:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
