import { useState, useEffect } from "react";
import axios from "axios";

function PredictionScreen() {
  const [latitude, setLatitude] = useState("");
  const [longitude, setLongitude] = useState("");
  const [predictPlace, setPredictPlace] = useState("");
  const [probability, setAverageProb] = useState(null);

  const handleLatitudeChange = (event) => {
    setLatitude(event.target.value);
  };

  const handleLongitudeChange = (event) => {
    setLongitude(event.target.value);
  };

  const handleSubmitClick = async () => {
    try {
      const lat = parseFloat(latitude);
      const lng = parseFloat(longitude);

      // Reverse geocode using OpenStreetMap (free)
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`
      );
      const data = await response.json();

      const locationName = data.display_name || "Unknown Location";
      setPredictPlace(locationName);

      // TODO: Call your backend here with lat/lng
      // const backendResponse = await fetch("/your-backend-api", {
      //     method: "POST",
      //     body: JSON.stringify({ lat, lng }),
      // });
      // const result = await backendResponse.json();
      // setProbability(result.probability);

      setLatitude("");
      setLongitude("");
    } catch (error) {
      console.error("Error:", error);
    }
  };

  useEffect(() => {
    const fetchPrediction = async () => {
      if (latitude && longitude) {
        try {
          const res = await axios.post(
            "http://localhost:5000/predictionUsingCurrentLocation",
            {
              latitude: latitude,
              longitude: longitude,
            }
          );
          const prediction = res.data.body;

          console.log(prediction);

          const probValues = Object.values(prediction.probabilities);
          const avg =
            probValues.reduce((sum, val) => sum + val, 0) / probValues.length;
          console.log(avg);
          setAverageProb(avg * 100);
        } catch (error) {
          console.error("Error fetching prediction:", error);
        }
      }
    };

    fetchPrediction();
  }, [latitude, longitude]);

  return (
    <div className="earthquake-prediction-cont">
      <p className="earthquake-prediction-title">Earthquake Predictor</p>
      <p className="earthquake-prediction-desc">
        Predict the earthquake and save your loved ones.
      </p>
      <div className="earthquake-prediction-form">
        <label className="earthquake-pred-label">Latitude</label>
        <input
          type="text"
          value={latitude}
          onChange={handleLatitudeChange}
          className="earthquake-pred-field"
        />
        <label className="earthquake-pred-label">Longitude</label>
        <input
          type="text"
          value={longitude}
          onChange={handleLongitudeChange}
          className="earthquake-pred-field"
        />
        <button onClick={handleSubmitClick} className="predict-btn">
          Start Prediction
        </button>
        {predictPlace ? (
          <>
            <p className="predict-place-title">You Searched For:</p>
            <p className="predict-place-name">{predictPlace}</p>
            <ProbabilityIndicator prob={probability} />
          </>
        ) : null}
      </div>
    </div>
  );
}

function ProbabilityIndicator({ prob }) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Animate from 0 to actual probability
    setTimeout(() => {
      setProgress(prob);
    }, 200); // delay to trigger animation
  }, [prob]);

  const radius = 120;
  const stroke = 10;
  const normalizedRadius = radius - stroke / 2;
  const circumference = 2 * Math.PI * normalizedRadius;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  // Gradient color from green to red
  const getColor = (p) => {
    if (p < 30) return "#4caf50"; // green
    if (p < 60) return "#ff9800"; // orange
    return "#f44336"; // red
  };

  return (
    <div className="prob-indicator">
      <svg height={radius * 2} width={radius * 2}>
        <circle
          stroke="#ddd"
          fill="transparent"
          strokeWidth={stroke}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
        />
        <circle
          stroke={getColor(progress)}
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference + " " + circumference}
          style={{
            strokeDashoffset,
            transition: "stroke-dashoffset 1s ease-out, stroke 0.5s ease-out",
          }}
          r={normalizedRadius}
          cx={radius}
          cy={radius}
          transform={`rotate(-90 ${radius} ${radius})`}
        />
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dy=".3em"
          fontSize="20px"
          fill="#333"
        >
          Probability {progress.toFixed(0)}%
        </text>
      </svg>
    </div>
  );
}
export default PredictionScreen;
