import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model and label encoders
MODEL_PATH = 'models/eurovision_gbr_model.pkl'
ENCODERS_PATH = 'models/label_encoders.pkl'

# Load model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Load label encoders
with open(ENCODERS_PATH, 'rb') as f:
    label_encoders = pickle.load(f)

# Define feature columns based on your training data
feature_columns = [
    'Year', 'Country', 'Region', 'Artist', 'Song', 'Artist.gender', 
    'Group.Solo', 'Place', 'Home.Away.Country', 'Home.Away.Region', 
    'Is.Final', 'Song.In.English', 'Song.Quality', 'Normalized.Points',
    'energy', 'duration', 'acousticness', 'danceability', 'tempo', 
    'speechiness', 'key', 'liveness', 'time_signature', 'mode', 
    'loudness', 'valence'
]

# Define categorical columns for encoding
categorical_cols = [
    'Country', 'Region', 'Artist', 'Song', 'Artist.gender', 'Group.Solo',
    'Home.Away.Country', 'Home.Away.Region'
]

# Define default values for numeric features
default_values = {
    'Year': 2024,
    'Is.Final': 1,
    'Song.In.English': 1,
    'Song.Quality': 0.5,
    'Normalized.Points': 0.1,
    'energy': 0.7,
    'duration': 180.0,
    'acousticness': 0.3,
    'danceability': 0.6,
    'tempo': 120.0,
    'speechiness': 0.05,
    'key': 5,
    'liveness': 0.2,
    'time_signature': 4,
    'mode': 1,
    'loudness': -8.0,
    'valence': 0.5,
    'Place': 10  # Default placement estimate
}

# Country options from your dataset
countries = [
    'Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium',
    'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
    'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece',
    'Hungary', 'Iceland', 'Ireland', 'Israel', 'Italy', 'Latvia', 'Lithuania',
    'Macedonia', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands',
    'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'San Marino', 'Serbia',
    'Serbia and Montenegro', 'Slovakia', 'Slovenia', 'Spain', 'Sweden',
    'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom'
]

regions = [
    'Former Socialist Bloc', 'Former Yugoslavia', 'Independent', 
    'Scandinavia', 'Western Europe'
]

artist_genders = ['Male', 'Female', 'Both', 'Unknown']
group_solo = ['Solo', 'Group', 'Unknown']
home_away = ['Home', 'Away']

@app.route('/')
def index():
    """Render the main prediction page"""
    return render_template('index.html',
                         countries=countries,
                         regions=regions,
                         artist_genders=artist_genders,
                         group_solo=group_solo,
                         home_away=home_away,
                         defaults=default_values)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Create a dictionary with all input values
        input_data = {}
        
        # Basic Information
        input_data['Year'] = int(request.form.get('Year', default_values['Year']))
        input_data['Country'] = request.form.get('Country', '')
        input_data['Region'] = request.form.get('Region', '')
        input_data['Artist'] = request.form.get('Artist', 'Unknown Artist')
        input_data['Song'] = request.form.get('Song', 'Unknown Song')
        input_data['Artist.gender'] = request.form.get('Artist.gender', 'Unknown')
        input_data['Group.Solo'] = request.form.get('Group.Solo', 'Unknown')
        
        # Contest Information
        input_data['Place'] = int(request.form.get('Place', default_values['Place']))
        input_data['Home.Away.Country'] = request.form.get('Home.Away.Country', 'Away')
        input_data['Home.Away.Region'] = request.form.get('Home.Away.Region', 'Away')
        input_data['Is.Final'] = int(request.form.get('Is.Final', default_values['Is.Final']))
        input_data['Song.In.English'] = int(request.form.get('Song.In.English', default_values['Song.In.English']))
        
        # Audio Features
        input_data['Song.Quality'] = float(request.form.get('Song.Quality', default_values['Song.Quality']))
        input_data['Normalized.Points'] = float(request.form.get('Normalized.Points', default_values['Normalized.Points']))
        input_data['energy'] = float(request.form.get('energy', default_values['energy']))
        input_data['duration'] = float(request.form.get('duration', default_values['duration']))
        input_data['acousticness'] = float(request.form.get('acousticness', default_values['acousticness']))
        input_data['danceability'] = float(request.form.get('danceability', default_values['danceability']))
        input_data['tempo'] = float(request.form.get('tempo', default_values['tempo']))
        input_data['speechiness'] = float(request.form.get('speechiness', default_values['speechiness']))
        input_data['key'] = int(request.form.get('key', default_values['key']))
        input_data['liveness'] = float(request.form.get('liveness', default_values['liveness']))
        input_data['time_signature'] = int(request.form.get('time_signature', default_values['time_signature']))
        input_data['mode'] = int(request.form.get('mode', default_values['mode']))
        input_data['loudness'] = float(request.form.get('loudness', default_values['loudness']))
        input_data['valence'] = float(request.form.get('valence', default_values['valence']))
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        # Apply label encoding to categorical columns
        for col in categorical_cols:
            if col in input_df.columns and col in label_encoders:
                try:
                    # Handle unseen labels by using the most frequent class (0)
                    if input_df[col].iloc[0] in label_encoders[col].classes_:
                        input_df[col] = label_encoders[col].transform(input_df[col])
                    else:
                        input_df[col] = 0  # Default to first class
                except:
                    input_df[col] = 0
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Make prediction
        predicted_points = model.predict(input_df)[0]
        
        # Calculate confidence based on prediction range
        # Max points in Eurovision can be up to ~500, but typical range is 0-300
        confidence = min(100, max(0, (predicted_points / 200) * 100))
        
        # Generate performance rating
        if predicted_points >= 150:
            rating = "Excellent - Top 3 contender!"
            color = "success"
        elif predicted_points >= 100:
            rating = "Good - Likely Top 10"
            color = "primary"
        elif predicted_points >= 50:
            rating = "Average - Could qualify"
            color = "warning"
        else:
            rating = "Below Average - May struggle to qualify"
            color = "danger"
        
        # Store input for display
        display_input = input_data.copy()
        
        return render_template('result.html',
                             predicted_points=round(predicted_points, 2),
                             confidence=round(confidence, 1),
                             rating=rating,
                             color=color,
                             input_data=display_input,
                             countries=countries)
    
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for JSON predictions"""
    try:
        data = request.get_json()
        
        # Create DataFrame from JSON
        input_df = pd.DataFrame([data])
        
        # Apply label encoding
        for col in categorical_cols:
            if col in input_df.columns and col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col])
                except:
                    input_df[col] = 0
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_columns]
        
        # Make prediction
        predicted_points = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'predicted_points': round(predicted_points, 2),
            'message': 'Prediction successful'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)