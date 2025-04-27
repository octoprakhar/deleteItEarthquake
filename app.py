from flask import Flask,jsonify,request
import requests
import pandas as pd
from io import StringIO
import re
from EarthquakePred import EarthquakePred
import os
import random
from flask_cors import CORS
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

## 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


## Response body format
'''
{
code: Int,
body: ResponseBOdy?,
meg: String
}
'''


app = Flask(__name__)
CORS(app)

# Global predictor object
predictor = None

# Initialize and train/load models only once on server start
with app.app_context():
    if not os.path.exists('models/stacking_model.pkl'):
        print("🚀 Training and saving models...")
        EarthquakePred.initialize_and_save_models()
    predictor = EarthquakePred.load_models()
    print("✅ Models ready.")


# Initialize and train models only once on server start
with app.app_context():
    if not os.path.exists('models/stacking_model.pkl'):
        print("🚀 Initializing models...")
        EarthquakePred.initialize_and_save_models()
    else:
        print("✅ Models already exist, skipping training.")

## For just testing
@app.route('/')
def home():
    return "Hello World"

@app.route('/test')
def test():
    number = random.randint(1, 100)  # You can change the range as needed
    return jsonify({"random": number})

## For emergency contact
@app.route("/emergencyContact", methods=['POST'])
def emergencyContact():
    '''
    request body {"name": String, "email": String, "subject": String, "message": String , "isImp": Boolean}
    '''
    SENDGRID_API_KEY = "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"  
    # TO_EMAIL = "kumarisonal0929@gmail.com"
    TO_EMAIL = "kumarisonal0929@gmail.com"
    data = request.get_json()
    print("Received Emergency Contact Form:", data)

    # Send email via SendGrid
    try:
        message = Mail(
            from_email='kumarisonalsingh2@gmail.com',  
            to_emails=TO_EMAIL,
            subject=f"Emergency Alert - {data.get('subject')}",
            html_content=f"""
                <strong>Name:</strong> {data.get('name')}<br>
                <strong>Email:</strong> {data.get('email')}<br>
                <strong>Important:</strong> {"Yes" if data.get("isImp") else "No"}<br>
                <strong>Message:</strong><br>{data.get('message')}
            """
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print("Email sent! Status Code:", response.status_code)

    except Exception as e:
        print("SendGrid Error:", e)
        return jsonify({"message": "Failed to send email", "error": str(e)}), 500

    return jsonify({
        "message": "Emergency contact received and email sent!",
        "data": data
    }), 200

## Prediction according current
@app.route('/predictionUsingCurrentLocation', methods=['POST'])
def predict_earthquake():
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')
        if lat is None or lon is None:
            return jsonify({'error': 'Missing latitude or longitude'}), 400

        result = predictor.get_earthquake_probability(lat, lon)
        print(result)
        return jsonify({'body': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route("/filterEarthquakes", methods=["POST"])
def filter_earthquakes():
    data = request.get_json()
    start_date = data.get("startDate")
    end_date = data.get("endDate")
    location = data.get("location")
    magnitude = data.get("magnitude")

    # Load your CSV file or dataset
    df = pd.read_csv("./data/usgsEarthquakeData.csv")

    # Ensure the 'time' column is datetime with UTC
    df['time'] = pd.to_datetime(df['time'], utc=True)

    # Convert input dates to UTC-aware datetime
    if start_date:
        start_date = pd.to_datetime(start_date).tz_localize("UTC")
        df = df[df['time'] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date).tz_localize("UTC")
        df = df[df['time'] <= end_date]

    if location:
        df = df[df['place'].str.contains(location, case=False, na=False)]

    if magnitude:
        df = df[df['mag'] >= float(magnitude)]

    # Return filtered data
    return jsonify(df.to_dict(orient="records"))



## Most affected regions
@app.route('/mostAffectedRegions', methods=['GET'])
def most_affected_regions():
# Load earthquake data
    data_path = os.path.join("data", "usgsEarthquakeData.csv")
    df = pd.read_csv(data_path)

    # Clean and extract region names from the 'place' column
    #  the format is like '20km SE of California'
    df['region'] = df['place'].apply(lambda x: x.split('of')[-1].strip() if 'of' in x else x.strip())

    # Count the number of earthquakes per region
    region_counts = df['region'].value_counts().head(10)

    # Format into desired JSON format
    result = [{region: int(count)} for region, count in region_counts.items()]

    return jsonify(result)


@app.route("/magnitudeWiseEarthquake", methods=['GET'])
def magnitude_wise_earthquake():
# Load earthquake data
    data_path = os.path.join("data", "usgsEarthquakeData.csv")
    df = pd.read_csv(data_path)

    # Round magnitudes to the nearest whole number (if needed)
    df['mag'] = df['mag'].round().astype(int)

    # Count frequency of each magnitude
    mag_counts = df['mag'].value_counts().sort_index()

    # Format into desired JSON format
    result = [{int(mag): int(count)} for mag, count in mag_counts.items()]

    return jsonify(result)


@app.route("/allEarthquakesDetails", methods=['GET'])
def all_earthquake_details():
# Load earthquake data
    data_path = os.path.join("data", "usgsEarthquakeData.csv")
    df = pd.read_csv(data_path)

    # Select only the required columns
    required_columns = ['latitude', 'longitude', 'mag', 'depth']

    # Drop rows with missing values in these columns (optional but cleaner)
    df = df[required_columns].dropna()

    # Convert to list of dictionaries
    result = df.to_dict(orient='records')

    return jsonify(result)


@app.route("/weatherInfo", methods=['POST'])
def weatherInfo():
    '''
    Request body:
    {
        "latitude": float,
        "longitude": float
    }

    Response body:
    {
        "code": int,
        "body": {
            "latitude": float,
            "longitude": float,
            "localtime": string,
            "temperature": float,
            "humidity": float,
            "precipitation": float,
            "feelsLike": float,
            "condition": string,
            "windSpeed": float
        },
        "msg": string
    }
    '''

    data = request.get_json()
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    # Validate input
    if latitude is None or longitude is None:
        return jsonify({
            "code": 101,
            "body": None,
            "msg": "Missing latitude or longitude"
        })

    # Format query string as "lat,lon"
    query = f"{latitude},{longitude}"

    try:
        response = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key=0bc5cb6b1927464f87993104251202&q={query}"
        )
        if response.status_code != 200:
            return jsonify({
                "code": 102,
                "body": None,
                "msg": "Failed to get weather data"
            })

        weather_data = response.json()
        return jsonify({
            "code": 1,
            "body": {
                "latitude": weather_data['location']['lat'],
                "longitude": weather_data['location']['lon'],
                "localtime": weather_data['location']['localtime'],
                "temperature": weather_data['current']['temp_c'],
                "humidity": weather_data['current']['humidity'],
                "precipitation": weather_data['current']['precip_mm'],
                "feelsLike": weather_data['current']["feelslike_c"],
                "condition": weather_data["current"]["condition"]["text"],
                "windSpeed": weather_data['current']["wind_kph"]
            },
            "msg": "Success"
        })

    except Exception as e:
        return jsonify({
            "code": 500,
            "body": None,
            "msg": f"Internal Server Error: {str(e)}"
        })






if __name__ == "__main__":
    app.run( debug=True) 