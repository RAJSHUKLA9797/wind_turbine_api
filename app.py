from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from scipy import stats
import math
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Helper Function to Check Allowed File Extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper Function to Calculate Mode for Continuous Data
def calculate_mode(data, bin_size=0.1):
    """
    Calculates the mode of a continuous dataset by creating a histogram.

    Parameters:
    - data (array-like): The data to calculate the mode for.
    - bin_size (float): The size of each bin.

    Returns:
    - mode (float): The mode value.
    """
    min_val = min(data)
    max_val = max(data)
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    hist, bin_edges = np.histogram(data, bins=bins)
    max_bin_index = np.argmax(hist)
    mode = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    return mode

# Function to Calculate Mb
def calculate_Mb(P):
    """
    Calculates Mb based on the power required.

    Parameters:
    - P (float): Power required in Megawatts (MW).

    Returns:
    - Mb (float): Calculated Mb value.
    """
    return 4343.7 * P - 420.23

# Function to Calculate Swept Area (A) and Diameter (D)
def calculate_swept_area_and_diameter(P, rho_mode, v_adjusted, Cp=0.4):
    """
    Calculates the swept area (A) and diameter (D) of the turbine.

    Parameters:
    - P (float): Power required in Megawatts (MW).
    - rho_mode (float): Mode of air density in kg/m³.
    - v_adjusted (float): Adjusted wind speed in m/s.
    - Cp (float): Power coefficient (default is 0.4).

    Returns:
    - A (float): Swept area in m².
    - D (float): Diameter in meters.
    """
    P_watts = P * 1e6  # Convert MW to W
    A = (2 * P_watts) / (rho_mode * (v_adjusted ** 3) * Cp)
    D = 2 * math.sqrt(A / math.pi)
    return A, D

# Function to Calculate Rcog
def calculate_Rcog(D):
    """
    Calculates Rcog based on the diameter.

    Parameters:
    - D (float): Diameter in meters.

    Returns:
    - Rcog (float): Calculated Rcog value in meters.
    """
    return 0.225 * D

# Function to Calculate Wn
def calculate_Wn(TSR, v_adjusted, D):
    """
    Calculates Wn based on the tip speed ratio, adjusted wind speed, and diameter.

    Parameters:
    - TSR (float): Tip Speed Ratio.
    - v_adjusted (float): Adjusted wind speed in m/s.
    - D (float): Diameter in meters.

    Returns:
    - Wn (float): Calculated Wn value in rad/s.
    """
    R = 0.5 * D  # Radius
    return TSR * v_adjusted / R

# Function to Calculate fzB
def calculate_fzB(Mb, Rcog, Wn):
    """
    Calculates the force fzB.

    Parameters:
    - Mb (float): Calculated Mb value.
    - Rcog (float): Calculated Rcog value in meters.
    - Wn (float): Calculated Wn value in rad/s.

    Returns:
    - fzB (float): Calculated force in Newtons (N).
    """
    return 2 * Mb * Rcog * (Wn ** 2)

# Function to Calculate Bending Moments ΔMxB and ΔMyB
def calculate_bending_moments(
    P_design,        # Design power in Watts (e.g., 1.5e6 for 1.5 MW)
    R,               # Rotor radius in meters
    V_avg,           # Average wind speed in m/s
    mb,              # Mass-related parameter in kg (e.g., blade mass)
    Rcog,            # Radius to center of gravity in meters
    lambda_design=7, # Design tip speed ratio (TSR), default is 7 for 3-blade turbine
    B=3,             # Number of blades, default is 3
    g=9.81           # Acceleration due to gravity in m/s², default is 9.81
):
    """
    Calculate the bending moments Delta MxB and Delta MyB for a wind turbine.
    
    Parameters:
    - P_design (float): Design power in Watts (W).
    - R (float): Rotor radius in meters (m).
    - V_avg (float): Average wind speed in meters per second (m/s).
    - mb (float): Mass-related parameter in kilograms (kg).
    - Rcog (float): Radius to center of gravity in meters (m).
    - lambda_design (float, optional): Design tip speed ratio (TSR). Default is 7.
    - B (int, optional): Number of blades. Default is 3.
    - g (float, optional): Acceleration due to gravity in m/s². Default is 9.81.
    
    Returns:
    - dict: A dictionary containing Delta_MxB and Delta_MyB in Newton-meters (Nm).
    """
    
    # Step 1: Calculate Design Wind Speed
    V_design = 1.5 * V_avg  # V_design = 1.5 * V_avg
    
    # Step 2: Calculate Design Rotational Speed (n_design) in RPM
    n_design = (lambda_design * V_design * 60) / (2 * math.pi * R)
    
    # Step 3: Calculate Design Angular Speed (omega_design) in rad/s
    omega_design = (2 * math.pi * n_design) / 60  # omega_design = 2π * n_design / 60
    
    # Step 4: Calculate Design Shaft Torque (Q_design) in Nm
    Q_design = P_design / omega_design  # Q_design = P_design / omega_design
    
    # Step 5: Calculate Bending Moment Delta_MxB
    Delta_MxB = (Q_design / B) + (2 * mb * g * Rcog)
    
    # Step 6: Calculate Bending Moment Delta_MyB
    Delta_MyB = lambda_design * (Q_design / B)
    
    # Return the results as a dictionary
    return {
        'Delta_MxB_Nm': Delta_MxB,
        'Delta_MyB_Nm': Delta_MyB
    }

# Main Calculation Function
def perform_calculations(P, data):
    """
    Performs all necessary calculations to determine fzB and bending moments.

    Parameters:
    - P (float): Power required in Megawatts (MW).
    - data (DataFrame): Prepared pandas DataFrame with 'rho' and 'wind_speed' columns.

    Returns:
    - results (dict): Dictionary containing all intermediate and final calculation results.
    """
    results = {}

    # ---------------------
    # NORMAL CALCULATIONS
    # ---------------------

    # Step 1: Calculate Mb
    Mb = calculate_Mb(P)
    results['normal'] = {}
    results['normal']['Mb'] = Mb

    # Step 2.1: Calculate the mode of rho
    rho_mode = calculate_mode(data['rho'])
    results['normal']['rho_mode'] = rho_mode

    # Step 2.2: Calculate the maximum wind speed and add 23.9%
    v_max = data['wind_speed'].max()
    v_adjusted = v_max * 1.239
    results['normal']['v_max'] = v_max
    results['normal']['v_adjusted'] = v_adjusted

    # Step 2.3 and 2.4: Calculate swept area A and diameter D
    A, D = calculate_swept_area_and_diameter(P, rho_mode, v_adjusted)
    results['normal']['A'] = A
    results['normal']['D'] = D

    # Step 2.5: Calculate Rcog
    Rcog = calculate_Rcog(D)
    results['normal']['Rcog'] = Rcog

    # Step 3.1 and 3.2: Calculate TSR=7, Wn, R
    TSR = 7  # Tip Speed Ratio for a 3-blade turbine
    Wn = calculate_Wn(TSR, v_adjusted, D)
    R = 0.5 * D
    results['normal']['TSR'] = TSR
    results['normal']['R'] = R
    results['normal']['Wn'] = Wn

    # Step 4: Calculate fzB
    fzB = calculate_fzB(Mb, Rcog, Wn)
    results['normal']['fzB'] = fzB

    # Step 5: Calculate Bending Moments Delta_MxB and Delta_MyB
    # Convert P from MW to Watts for the function
    P_watts = P * 1e6
    bending_moments = calculate_bending_moments(
        P_design=P_watts,
        R=R,
        V_avg=v_adjusted,
        mb=Mb,
        Rcog=Rcog,
        lambda_design=TSR,
        B=3,
        g=9.81
    )
    results['normal']['Delta_MxB'] = bending_moments['Delta_MxB_Nm']
    results['normal']['Delta_MyB'] = bending_moments['Delta_MyB_Nm']

    # ---------------------
    # LOAD CASE E CALCULATIONS
    # ---------------------

    # Step 1: Calculate nmax = TSR * v_adjusted / R
    nmax = TSR * v_adjusted / R
    results['max'] = {}
    results['max']['nmax'] = nmax

    # Step 2: Calculate ω_n,max = (π /30)*nmax
    omega_n_max = nmax
    results['max']['omega_n_max'] = omega_n_max

    # Step 3: Calculate fzB_max = mB * ω_n_max^2 * Rcog
    fzB_max = Mb * (omega_n_max ** 2) * Rcog
    results['max']['fzB_max'] = fzB_max

    # Step 4: Calculate mr = mB + mhub, where mhub =0.1 *3*mB=0.3 mB
    mhub = 0.3 * Mb
    mr = Mb + mhub
    results['max']['mhub'] = mhub
    results['max']['mr'] = mr

    # Step 5: Calculate Lrb =0.125 * R
    Lrb = 0.125 * R
    results['max']['Lrb'] = Lrb

    # Step 6: Calculate er=0.005 * R
    er = 0.005 * R
    results['max']['er'] = er

    # Step 7: Calculate Msha = mr * g * Lrb + mr * er * omega_n_max^2
    Msha = mr * 9.81 * Lrb + mr * er * (omega_n_max ** 2)
    results['max']['Msha'] = Msha

    return results

@app.route('/', methods=['GET'])
def index():
    """
    Renders the index page with the input form.
    """
    return render_template('index.html')

@app.route('/calculate_fzb', methods=['POST'])
def calculate_fzb():
    """
    Handles the form submission, performs calculations, and renders the results page.

    Expects:
    - 'power' (float): Power required in Megawatts (MW).
    - 'file' (file): CSV file with required columns.

    Returns:
    - Rendered results page with calculation outcomes or error messages.
    """
    # Check if 'power' is in form data
    if 'power' not in request.form:
        error = "Missing 'power' parameter."
        return render_template('results.html', error=error)

    # Check if 'file' is in files
    if 'file' not in request.files:
        error = "Missing 'file' parameter."
        return render_template('results.html', error=error)

    power = request.form.get('power')

    # Validate power input
    try:
        P = float(power)
        if P <= 0:
            error = "'power' must be a positive number."
            return render_template('results.html', error=error)
    except ValueError:
        error = "'power' must be a numerical value."
        return render_template('results.html', error=error)

    file = request.files['file']

    # Check if the file is selected and allowed
    if file.filename == '':
        error = "No selected file."
        return render_template('results.html', error=error)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Read CSV file
            data = pd.read_csv(filepath)

            # Check for required columns
            required_columns = ['Date', 'Year', 'Month', 'Density (rho) [kg/m³]', 'Wind Speed [m/s]']
            if not all(col in data.columns for col in required_columns):
                error = f"CSV file must contain the following columns: {required_columns}"
                return render_template('results.html', error=error)

            # Rename columns for easier access
            data.rename(columns={
                'Density (rho) [kg/m³]': 'rho',
                'Wind Speed [m/s]': 'wind_speed'
            }, inplace=True)

            # Ensure there are no missing values in required columns
            if data[['rho', 'wind_speed']].isnull().any().any():
                error = "CSV file contains missing values in 'Density (rho) [kg/m³]' or 'Wind Speed [m/s]' columns."
                return render_template('results.html', error=error)

            # Perform calculations
            results = perform_calculations(P, data)

            # Prepare data for rendering (format numbers)
            formatted_results = {
                # Normal Calculations
                'Mb': f"{results['normal']['Mb']:.2f}",
                'rho_mode': f"{results['normal']['rho_mode']:.4f}",
                'v_max': f"{results['normal']['v_max']:.2f}",
                'v_adjusted': f"{results['normal']['v_adjusted']:.2f}",
                'A': f"{results['normal']['A']:.2f}",
                'D': f"{results['normal']['D']:.2f}",
                'Rcog': f"{results['normal']['Rcog']:.2f}",
                'TSR': f"{results['normal']['TSR']:.2f}",
                'R': f"{results['normal']['R']:.2f}",
                'Wn': f"{results['normal']['Wn']:.4f}",
                'fzB': f"{results['normal']['fzB']:.2f}",
                'Delta_MxB': f"{results['normal']['Delta_MxB']:.2f}",
                'Delta_MyB': f"{results['normal']['Delta_MyB']:.2f}",
                # Load Case E Calculations
                'nmax': f"{results['max']['nmax']:.2f}",
                'omega_n_max': f"{results['max']['omega_n_max']:.4f}",
                'fzB_max': f"{results['max']['fzB_max']:.2f}",
                'mr': f"{results['max']['mr']:.2f}",
                'Lrb': f"{results['max']['Lrb']:.2f}",
                'er': f"{results['max']['er']:.2f}",
                'Msha': f"{results['max']['Msha']:.2f}"
            }

            return render_template('results.html', results=formatted_results)

        except Exception as e:
            error = f"An error occurred while processing the file: {str(e)}"
            return render_template('results.html', error=error)

        finally:
            # Remove the uploaded file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
    else:
        error = "Invalid file type. Only CSV files are allowed."
        return render_template('results.html', error=error)

if __name__ == '__main__':
    app.run(debug=True)