from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import math
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_mode(data, bin_size=0.1):
    min_val = min(data)
    max_val = max(data)
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    hist, bin_edges = np.histogram(data, bins=bins)
    max_bin_index = np.argmax(hist)
    mode = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    return mode

def calculate_Mb(P):
    return 4343.7 * P - 420.23

def calculate_swept_area_and_diameter(P, rho_mode, v_adjusted, Cp=0.4):
    P_watts = P * 1e6
    A = (2 * P_watts) / (rho_mode * (v_adjusted ** 3) * Cp)
    D = 2 * math.sqrt(A / math.pi)
    return A, D

def calculate_Rcog(D):
    return 0.225 * D

def calculate_Wn(TSR, v_adjusted, D):
    R = 0.5 * D
    return TSR * v_adjusted / R

def calculate_fzB(Mb, Rcog, Wn):
    return 2 * Mb * Rcog * (Wn ** 2)

def calculate_bending_moments(P_design, R, V_avg, mb, Rcog, lambda_design, B=3, g=9.81):
    V_design = 1.5 * V_avg
    n_design = (lambda_design * V_design * 60) / (2 * math.pi * R)
    omega_design = (2 * math.pi * n_design) / 60
    Q_design = P_design / omega_design
    Delta_MxB = (Q_design / B) + (2 * mb * g * Rcog)
    Delta_MyB = lambda_design * (Q_design / B)
    return {'Delta_MxB_Nm': Delta_MxB, 'Delta_MyB_Nm': Delta_MyB}

def perform_calculations(P, data, Cp=0.4, B=3):
    results = {}
    Mb = calculate_Mb(P)
    results['normal'] = {'Mb': Mb}
    rho_mode = calculate_mode(data['rho'])
    v_max = data['wind_speed'].max()
    v_adjusted = v_max * 1.239
    A, D = calculate_swept_area_and_diameter(P, rho_mode, v_adjusted, Cp)
    Rcog = calculate_Rcog(D)

    TSR = 15 / B
    lambda_design = TSR
    Wn = calculate_Wn(TSR, v_adjusted, D)
    R = 0.5 * D
    fzB = calculate_fzB(Mb, Rcog, Wn)
    P_watts = P * 1e6
    bending_moments = calculate_bending_moments(P_watts, R, v_adjusted, Mb, Rcog, lambda_design, B)

    results['normal'].update({
        'rho_mode': rho_mode,
        'v_max': v_max,
        'v_adjusted': v_adjusted,
        'A': A,
        'D': D,
        'Rcog': Rcog,
        'TSR': TSR,
        'R': R,
        'Wn': Wn,
        'fzB': fzB,
        'Delta_MxB': bending_moments['Delta_MxB_Nm'],
        'Delta_MyB': bending_moments['Delta_MyB_Nm']
    })

    nmax = TSR * v_adjusted / R
    omega_n_max = nmax
    fzB_max = Mb * (omega_n_max ** 2) * Rcog
    mhub = 0.3 * Mb
    mr = Mb + mhub
    Lrb = 0.125 * R
    er = 0.005 * R
    Msha = mr * 9.81 * Lrb + mr * er * (omega_n_max ** 2)

    results['max'] = {
        'nmax': nmax,
        'omega_n_max': omega_n_max,
        'fzB_max': fzB_max,
        'mhub': mhub,
        'mr': mr,
        'Lrb': Lrb,
        'er': er,
        'Msha': Msha
    }

    # Load Case H
    results['H'] = {}
    rho_50 = rho_mode
    zhub = R
    Vref = v_adjusted
    Ve50 = 1.4 * Vref * ((R / zhub) ** 0.11)
    c_avg = 0.08
    Aproj_B = R * c_avg
    CI_max = 2.0
    MyB = CI_max * (1/6) * rho_50 * (Ve50 ** 2) * Aproj_B * R
    lambda_e50 = (nmax * math.pi * R) / (30 * Ve50)
    Fx_sha = 0.17 * B * Aproj_B * (lambda_e50 ** 2) * rho_50 * (Ve50 ** 2)
    results['H'].update({
        'Ve50': Ve50,
        'Aproj_B': Aproj_B,
        'lambda_e50': lambda_e50,
        'MyB': MyB,
        'Fx_sha': Fx_sha
    })

    # Load Case I
    results['I'] = {}
    Cf_nacelle = 1.2
    Cf_tower = 1.2
    Cf_blade = 1.4
    Aproj_nacelle = 6.0
    Aproj_tower = 8.0
    Aproj_blade = Aproj_B * B

    F_nacelle = Cf_nacelle * 0.5 * rho_50 * Ve50**2 * Aproj_nacelle
    F_tower = Cf_tower * 0.5 * rho_50 * Ve50**2 * Aproj_tower
    F_blade = Cf_blade * 0.5 * rho_50 * Ve50**2 * Aproj_blade

    results['I'].update({
        'F_nacelle': F_nacelle,
        'F_tower': F_tower,
        'F_blade': F_blade
    })

    return results

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/calculate_fzb', methods=['POST'])
def calculate_fzb():
    if 'power' not in request.form or 'file' not in request.files:
        return render_template('results.html', error="Missing required inputs.")

    try:
        P = float(request.form.get('power'))
        if P <= 0:
            raise ValueError
    except ValueError:
        return render_template('results.html', error="'power' must be a positive number.")

    try:
        B = int(request.form.get('blades', 3))
    except ValueError:
        B = 3

    try:
        Cp = float(request.form.get('cp', 0.4))
    except ValueError:
        Cp = 0.4

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('results.html', error="Invalid or missing CSV file.")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        data = pd.read_csv(filepath)
        required_columns = ['Date', 'Year', 'Month', 'Density (rho) [kg/m³]', 'Wind Speed [m/s]']
        if not all(col in data.columns for col in required_columns):
            return render_template('results.html', error=f"CSV file must contain: {required_columns}")
        data.rename(columns={'Density (rho) [kg/m³]': 'rho', 'Wind Speed [m/s]': 'wind_speed'}, inplace=True)
        if data[['rho', 'wind_speed']].isnull().any().any():
            return render_template('results.html', error="CSV file contains missing values.")

        results = perform_calculations(P, data, Cp, B)
        formatted_results = {
            **{k: f"{v:.2f}" if isinstance(v, float) else v for k, v in {
                'Mb': results['normal']['Mb'],
                'rho_mode': results['normal']['rho_mode'],
                'v_max': results['normal']['v_max'],
                'v_adjusted': results['normal']['v_adjusted'],
                'A': results['normal']['A'],
                'D': results['normal']['D'],
                'Rcog': results['normal']['Rcog'],
                'TSR': results['normal']['TSR'],
                'R': results['normal']['R'],
                'Wn': results['normal']['Wn'],
                'fzB': results['normal']['fzB'],
                'Delta_MxB': results['normal']['Delta_MxB'],
                'Delta_MyB': results['normal']['Delta_MyB'],
                'nmax': results['max']['nmax'],
                'omega_n_max': results['max']['omega_n_max'],
                'fzB_max': results['max']['fzB_max'],
                'mr': results['max']['mr'],
                'Lrb': results['max']['Lrb'],
                'er': results['max']['er'],
                'Msha': results['max']['Msha'],
                'Ve50': results['H']['Ve50'],
                'Aproj_B': results['H']['Aproj_B'],
                'lambda_e50': results['H']['lambda_e50'],
                'MyB': results['H']['MyB'],
                'Fx_sha': results['H']['Fx_sha'],
                'F_nacelle': results['I']['F_nacelle'],
                'F_tower': results['I']['F_tower'],
                'F_blade': results['I']['F_blade']
            }.items()},
            'B': B,
            'Cp': Cp
        }
        return render_template('results.html', results=formatted_results)

    except Exception as e:
        return render_template('results.html', error=str(e))

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)