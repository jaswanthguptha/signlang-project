import os
import sys
import traceback
import numpy as np
import requests
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

def run_tests():
    results = {}
    
    # Step 1: Flask + numpy
    try:
        import numpy
        results['Step 1 (Flask + numpy)'] = {'status': 'SUCCESS'}
    except Exception as e:
        results['Step 1 (Flask + numpy)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}

    # Step 2: Add cv2
    try:
        import cv2
        results['Step 2 (cv2)'] = {'status': 'SUCCESS'}
    except Exception as e:
        results['Step 2 (cv2)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}

    # Step 3: Add mediapipe
    try:
        import mediapipe
        results['Step 3 (mediapipe)'] = {'status': 'SUCCESS'}
    except Exception as e:
        results['Step 3 (mediapipe)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}

    # Step 4: Add onnxruntime
    try:
        import onnxruntime
        results['Step 4 (onnxruntime)'] = {'status': 'SUCCESS'}
    except Exception as e:
        results['Step 4 (onnxruntime)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}

    # Step 5: Global model loading
    try:
        import pickle
        with open('models/model_alpha.pkl', 'rb') as f:
            pickle.load(f)
        results['Step 5 (Global model loading)'] = {'status': 'SUCCESS'}
    except Exception as e:
        results['Step 5 (Global model loading)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}
        
    # Send results to kvdb.io
    try:
        requests.post('https://kvdb.io/signlang_diag_7459/results', json=results, timeout=10)
        print('Diagnostics sent to kvdb.io successfully!')
    except Exception as ex:
        print('Failed to send diagnostics to kvdb.io:', ex)

if __name__ == '__main__':
    run_tests()
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  Server starting on http://0.0.0.0:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
