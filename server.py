import os
import sys
import traceback
import numpy as np
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

def run_tests():
    results = {}
    
    # Step 1: Flask + numpy
    print('--- STEP 1: Starting Flask and numpy ---')
    try:
        import numpy
        results['Step 1 (Flask + numpy)'] = {'status': 'SUCCESS'}
        print('Step 1 PASSED')
    except Exception as e:
        results['Step 1 (Flask + numpy)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}
        print('Step 1 FAILED:', e)

    # Step 2: Add cv2
    print('--- STEP 2: Adding cv2 ---')
    try:
        import cv2
        results['Step 2 (cv2)'] = {'status': 'SUCCESS'}
        print('Step 2 PASSED')
    except Exception as e:
        results['Step 2 (cv2)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}
        print('Step 2 FAILED:', e)

    # Step 3: Add mediapipe
    print('--- STEP 3: Adding mediapipe ---')
    try:
        import mediapipe
        results['Step 3 (mediapipe)'] = {'status': 'SUCCESS'}
        print('Step 3 PASSED')
    except Exception as e:
        results['Step 3 (mediapipe)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}
        print('Step 3 FAILED:', e)

    # Step 4: Add onnxruntime
    print('--- STEP 4: Adding onnxruntime ---')
    try:
        import onnxruntime
        results['Step 4 (onnxruntime)'] = {'status': 'SUCCESS'}
        print('Step 4 PASSED')
    except Exception as e:
        results['Step 4 (onnxruntime)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}
        print('Step 4 FAILED:', e)

    # Step 5: Global model loading
    print('--- STEP 5: Global model loading ---')
    try:
        import pickle
        with open('models/model_alpha.pkl', 'rb') as f:
            pickle.load(f)
        results['Step 5 (Global model loading)'] = {'status': 'SUCCESS'}
        print('Step 5 PASSED')
    except Exception as e:
        results['Step 5 (Global model loading)'] = {'status': 'FAILURE', 'exception': traceback.format_exc()}
        print('Step 5 FAILED:', e)
        
    print('\n=== STARTUP ISOLATION TEST RESULTS ===')
    for step, res in results.items():
        print(f'{step}: {res['status']}')
        if res['status'] == 'FAILURE':
            print(res['exception'])
    print('======================================\n')

if __name__ == '__main__':
    run_tests()
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  Server starting on http://0.0.0.0:{port}\n')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
