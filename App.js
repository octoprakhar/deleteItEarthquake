import "./App.css";
import { useEffect, useState } from "react";
import axios from "axios";

import { collection, addDoc, doc } from "firebase/firestore";
import { onAuthStateChanged } from "firebase/auth";
import { auth, db } from "./firebase";
import { getDoc, query, where, getDocs } from "firebase/firestore";

import IndiaMap from "./components/IndiaMap";
import RegisterUser from "./components/RegisterUser";
import LoginUser from "./components/SignInUser";
import ContactPage from "./components/ContactScreen";
import PreparednessScreen from "./components/PreparednessScreen";
import NewsAndUpdatesScreen from "./components/NewsAndUpdateScreen";
import HistoricalDataScreen from "./components/HistoricalDataScreen";
import MapScreen from "./components/MapScreen";
import PredictionScreen from "./components/PredictionScreen";

function App() {
  const [active, setactive] = useState("home");

  const [earthquakeData, setEarthquakeData] = useState();
  const [position, setPosition] = useState();

  const [user, setUser] = useState(null); //Check user is present or not

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
        try {
          const q = query(
            collection(db, "users"),
            where("uid", "==", currentUser.uid)
          );
          const snapshot = await getDocs(q);
          if (!snapshot.empty) {
            const userData = snapshot.docs[0].data();
            const { latitude, longitude } = userData;
            setPosition({ lat: latitude, lng: longitude });
          } else {
            console.log("User data not found in Firestore");
          }
        } catch (error) {
          console.error("Error fetching user doc:", error);
        }
      } else {
        setUser(null);
        setPosition(null);
      }
    });

    return () => unsubscribe();
  }, []);

  const handleActiveChange = (newActive) => {
    setactive(newActive);
  };

  const handleEarthquakeData = (newData) => {
    setEarthquakeData(newData);
  };

  useEffect(() => {
    const fetchEarthquakeData = async () => {
      try {
        const response = await axios.get(
          "http://localhost:5000/allEarthquakesDetails"
        );
        console.log(response.data);
        setEarthquakeData(response.data);
      } catch (error) {
        console.error("Error fetching earthquake data:", error);
      }
    };

    if (active === "map") {
      fetchEarthquakeData();
    }
  }, [active]);

  return (
    <>
      <div className="main-cont">
        <div className="main-nav-cont">
          <MainNav
            props={{
              active: active,
              handleActiveChange: handleActiveChange,
              user: user,
            }}
          />
        </div>
      </div>
      {active === "home" ? (
        <>
          <HomePoster
            navigateToPrediction={() => handleActiveChange("prediction")}
            navigateToPrepare={() => handleActiveChange("preparedness")}
          />
          <HomePosterExtra position={position} />
          <EarthquakePrediction />
        </>
      ) : active === "register" ? (
        <RegisterUser
          onLoginClick={handleActiveChange}
          setPosition={setPosition}
        />
      ) : active === "login" ? (
        <LoginUser
          onRegisterClick={handleActiveChange}
          setPosition={setPosition}
        />
      ) : active === "contact" ? (
        <ContactPage />
      ) : active === "preparedness" ? (
        <PreparednessScreen />
      ) : active === "news" ? (
        <NewsAndUpdatesScreen position={position} />
      ) : active === "historical data" ? (
        <HistoricalDataScreen />
      ) : active === "map" ? (
        <MapScreen datapoint={earthquakeData} />
      ) : active === "prediction" ? (
        <PredictionScreen />
      ) : (
        <h1>Please wait we are making this page.</h1>
      )}
    </>
  );
}

function MainNav({ props }) {
  return (
    <>
      <div className="app-name">
        <strong>EarthQuakeShield</strong>
      </div>
      <nav className="main-nav">
        {!props.user && (
          <>
            <button onClick={() => props.handleActiveChange("register")}>
              Register
            </button>
            <button onClick={() => props.handleActiveChange("login")}>
              Login
            </button>
          </>
        )}
        {props.user && (
          <ul className="nav-links">
            <li
              className={props.active === "home" ? "active-nav" : ""}
              onClick={() => props.handleActiveChange("home")}
            >
              Home
            </li>
            <li
              className={props.active === "map" ? "active-nav" : ""}
              onClick={() => props.handleActiveChange("map")}
            >
              Map
            </li>
            <li
              className={props.active === "historical data" ? "active-nav" : ""}
              onClick={() => props.handleActiveChange("historical data")}
            >
              Historical Data
            </li>
            <li
              className={props.active === "prediction" ? "active-nav" : ""}
              onClick={() => props.handleActiveChange("prediction")}
            >
              Prediction
            </li>
            <li
              className={props.active === "preparedness" ? "active-nav" : ""}
              onClick={() => props.handleActiveChange("preparedness")}
            >
              Preparedness
            </li>
            <li
              className={props.active === "contact" ? "active-nav" : ""}
              onClick={() => props.handleActiveChange("contact")}
            >
              Contact
            </li>
            <li
              className={props.active === "news" ? "active-nav" : ""}
              onClick={() => props.handleActiveChange("news")}
            >
              News
            </li>
            <li
              className={props.active === "Logout" ? "active-nav" : ""}
              onClick={() => auth.signOut()}
            >
              Logout
            </li>
          </ul>
        )}
      </nav>
    </>
  );
}

function HomePoster({ navigateToPrediction, navigateToPrepare }) {
  return (
    <>
      <div className="home-poster-cont">
        <div className="home-poster-text">
          <h1 className="home-poster-title">Predict. Prepare. Protect.</h1>
          <p className="home-poster-desc">
            Advanced earthquake prediction and disaster management system to
            help communities stay safe and informed.
          </p>
          <div className="home-poster-btn">
            <button
              type="Submit"
              className="Prediction-btn"
              onClick={navigateToPrediction}
            >
              View Predictions
            </button>
            <button
              type="Submit"
              className="Contact-btn"
              onClick={navigateToPrepare}
            >
              Contact Emergency Services
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

function HomePosterExtra({ position }) {
  const [locationText, setLocationText] = useState("Fetching...");

  useEffect(() => {
    if (position) {
      calculateaddressFromLatLng(position);
    }
  }, [position]);

  const calculateaddressFromLatLng = async (position) => {
    try {
      const { lat, lng } = position;
      const response = await fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=10&addressdetails=1`
      );
      const data = await response.json();
      const address = data.address;
      const state = address.state || "N/A";
      const country = address.country || "N/A";
      setLocationText(`${state}, ${country}`);
    } catch (error) {
      console.error("Error getting location:", error);
      setLocationText("Unable to determine location");
    }
  };

  return (
    <div className="home-poster-extra-cont">
      <div className="detail1">
        <img
          src="/alert.svg"
          height="50px"
          width="50px"
          className="detail1-img"
        />
        <div className="details">
          <p className="detail1-title">Active Alert</p>
          <p>15</p>
        </div>
      </div>
      <div className="detail2">
        <img
          src="/globe.svg"
          height="50px"
          width="50px"
          className="detail2-img"
        />
        <div className="details">
          <p className="detail1-title">Recent earthquake(24hrs)</p>
          <p>31</p>
        </div>
      </div>
      <div className="detail3">
        <img
          src="/gps.svg"
          height="50px"
          width="50px"
          className="detail3-img"
        />
        <div className="details">
          <p className="detail1-title">Your Location</p>
          <p>{locationText}</p>
        </div>
      </div>
    </div>
  );
}

function RegionalDetail({ props }) {
  return (
    <div className="regional-detail-cont">
      <div className="region-name">
        <p>{props.name}</p>
        <div
          className={
            props.riskIndicator === 1 ? "region-ind-mod" : "region-ind"
          }
        >
          {props.riskIndicator === 0
            ? "Mild"
            : props.riskIndicator === 1
            ? "Moderate"
            : props.riskIndicator === 2
            ? "High Risk"
            : "Very High"}
        </div>
      </div>
      <p className="normal-para">{props.place}</p>
      <hr
        className={
          props.riskIndicator === 1 ? "regional-line-mod" : "regional-line"
        }
      />
      <p className="normal-para">{props.desc}</p>
    </div>
  );
}

function EarthquakePrediction() {
  const [lat, setLat] = useState(null);
  const [lng, setLng] = useState(null);
  const [magnitude, setMagnitude] = useState(null);
  const [riskLevel, setRiskLevel] = useState(null);
  const [averageProb, setAverageProb] = useState(null);

  useEffect(() => {
    const fetchUserLocation = async () => {
      const user = auth.currentUser;
      if (user) {
        const q = query(collection(db, "users"), where("uid", "==", user.uid));
        const snapshot = await getDocs(q);
        if (!snapshot.empty) {
          const data = snapshot.docs[0].data();
          setLat(data.latitude);
          setLng(data.longitude);
        }
      }
    };

    fetchUserLocation();
  }, []);

  useEffect(() => {
    const fetchPrediction = async () => {
      if (lat && lng) {
        try {
          const res = await axios.post(
            "http://localhost:5000/predictionUsingCurrentLocation",
            {
              latitude: lat,
              longitude: lng,
            }
          );
          const prediction = res.data.body;
          setMagnitude(prediction.predicted_magnitude);
          setRiskLevel(prediction.risk_level);

          const probValues = Object.values(prediction.probabilities);
          const avg =
            probValues.reduce((sum, val) => sum + val, 0) / probValues.length;
          setAverageProb(avg);
        } catch (error) {
          console.error("Error fetching prediction:", error);
        }
      }
    };

    fetchPrediction();
  }, [lat, lng]);

  return (
    <div className="prediction-cont">
      <svg
        className="prediction-icon"
        fill="#f50505"
        viewBox="-1.28 -1.28 34.56 34.56"
        width="70px"
        height="70px"
        xmlns="http://www.w3.org/2000/svg"
      >
        <g
          id="SVGRepo_bgCarrier"
          strokeWidth="0"
          transform="translate(0,0), scale(1)"
        >
          <rect
            x="-1.28"
            y="-1.28"
            width="34.56"
            height="34.56"
            rx="17.28"
            fill="#dd928d"
          />
        </g>
        <g
          id="SVGRepo_tracerCarrier"
          strokeLinecap="round"
          strokeLinejoin="round"
          stroke="#CCCCCC"
          strokeWidth="0.064"
        />
        <g id="SVGRepo_iconCarrier">
          <title>heartbeat</title>
          <path d="M10.24 23.84c-0.040 0-0.040 0-0.080 0-0.4-0.040-0.72-0.36-0.76-0.76l-1-9.96-0.88 2.72c-0.12 0.36-0.44 0.56-0.8 0.56h-5.88c-0.48 0-0.84-0.36-0.84-0.8s0.36-0.84 0.84-0.84h5.32l1.92-6c0.12-0.36 0.48-0.6 0.88-0.56s0.68 0.36 0.72 0.76l0.96 9.72 2.040-6.96c0.12-0.36 0.4-0.6 0.8-0.6 0.36 0 0.68 0.28 0.8 0.64l1.12 4.84 0.84-1.24c0.16-0.24 0.4-0.36 0.68-0.36h5.28c0.44 0 0.84 0.36 0.84 0.84 0 0.44-0.36 0.84-0.84 0.84h-4.84l-1.64 2.44c-0.2 0.28-0.52 0.4-0.84 0.36s-0.6-0.32-0.64-0.64l-0.84-3.6-2.36 8.080c-0.12 0.28-0.44 0.52-0.8 0.52z"></path>
        </g>
      </svg>
      <p className="prediction-title">Earthquake Prediction & Analysis</p>
      <p className="prediction-desc">
        Using advanced algorithms and real time seismic data to provide accurate
        earthquake predictions and risk assessments.
      </p>
      <div className="prediction-body">
        <div className="prediction-map-cont">
          <p>Current Prediction Map</p>
          <div className="prediction-map">
            <IndiaMap lat={lat} lng={lng} />
          </div>
          <div className="prediction-result-container">
            <div className="prediction-result1">
              <p>Risk Level</p>
              <p>{riskLevel ? riskLevel : "Load..."}</p>
            </div>
            <div className="prediction-result2">
              <p>Probability</p>
              <p>{averageProb ? averageProb.toFixed(2) * 100 : "Load..."} %</p>
            </div>
            <div className="prediction-result3">
              <p>Potential Magnitude</p>
              <p>{magnitude ? magnitude.toFixed(2) : "Load..."}</p>
            </div>
            <div className="prediction-result4">
              <p>Time Frame</p>
              <p>Next 10 days</p>
            </div>
          </div>
        </div>
        <div className="prediction-table-cont">
          <p className="prediction-table-title">Regional Risk Assessment</p>
          <RegionalDetail
            props={{
              id: 0,
              riskIndicator: 2,
              percent: 60,
              name: "Northern India Himalayan Belt",
              place: "Uttrakhand,India",
              desc: "72 percent Probability within next 30 days.",
            }}
          />
          <RegionalDetail
            props={{
              id: 0,
              riskIndicator: 3,
              percent: 60,
              name: "Northeast Belt",
              place: "Assam,India",
              desc: "85 percent Probability within next 30 days.",
            }}
          />
          <RegionalDetail
            props={{
              id: 0,
              riskIndicator: 1,
              percent: 60,
              name: "Gujarat Kachh Region",
              place: "Western India",
              desc: "49 percent Probability within next 30 days.",
            }}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
