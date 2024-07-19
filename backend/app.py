from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from backend.mmr_power import powfun

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/calculate_power', methods=['POST'])
def calculate_power():
    data = request.json

    required_keys = ['alpha', 'k', 'n', 'sigx_obs', 'sigy_obs', 'alphax', 'alphay', 'rho_obs', 'TruncProp']
    for key in required_keys:
        if key not in data:
            return jsonify({'error': f'Missing required parameter: {key}'}), 400

    try:
        Nmult = 1
        max_power = []
        
        alpha = data['alpha']
        k = data['k']
        n = data['n']
        sigxobs = data['sigx_obs']
        sigyobs = data['sigy_obs']
        alphax = data['alphax']
        alphay = data['alphay']
        rhoobs = data['rho_obs']
        TruncProp = data['TruncProp']

        power, chsq_mean, chsq_var = powfun(Nmult, max_power, alpha, k, n, sigxobs, sigyobs, alphax, alphay, rhoobs, TruncProp)
        # Convert NaN to None to ensure valid JSON response
        if np.isnan(power):
            power = None
            
        print('Calculated power: ', power)
        return jsonify({'power': power})
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
