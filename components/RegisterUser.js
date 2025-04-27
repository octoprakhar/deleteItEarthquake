import "../App.css";
import { useState } from "react";

import { createUserWithEmailAndPassword } from "firebase/auth";
import { collection, addDoc } from "firebase/firestore";
import { auth, db } from "../firebase";

function RegisterUser({ onLoginClick, setPosition }) {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [password, setPassword] = useState("");
  const [latitude, setLatitude] = useState(null);
  const [longitude, setLongitude] = useState(null);

  const getLocation = () => {
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) {
        reject(new Error("Geolocation is not supported by your browser"));
      } else {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            resolve({
              lat: position.coords.latitude,
              lng: position.coords.longitude,
            });
          },
          (error) => {
            reject(error);
          }
        );
      }
    });
  };

  const handleFirstNameChange = (event) => {
    setFirstName(event.target.value);
  };

  const handleLastNameChange = (event) => {
    setLastName(event.target.value);
  };

  const handleEmailChange = (event) => {
    setEmail(event.target.value);
  };

  const handlePhoneNumberChange = (event) => {
    setPhoneNumber(event.target.value);
  };

  const handlePasswordChange = (event) => {
    setPassword(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      // Ask for location
      const location = await getLocation();
      setLatitude(location.lat);
      setLongitude(location.lng);

      setPosition({ lat: location.lat, lng: location.lng });

      // Create user with email and password
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email,
        password
      );
      const user = userCredential.user;

      // Add other details to Firestore
      await addDoc(collection(db, "users"), {
        uid: user.uid,
        firstName,
        lastName,
        email,
        password,
        phoneNumber,
        latitude: location.lat,
        longitude: location.lng,
      });

      console.log("User registered successfully!");
      setFirstName("");
      setLastName("");
      setEmail("");
      setPhoneNumber("");
      setPassword("");
    } catch (error) {
      console.error("Error registering user:", error.message);
      setFirstName("");
      setLastName("");
      setEmail("");
      setPhoneNumber("");
      setPassword("");
    }
  };

  return (
    <div className="register-cont">
      <p className="register-title">Register Yourself Here.</p>
      <form className="register-form" onSubmit={handleSubmit}>
        <div className="form-row">
          <label htmlFor="fname">First Name:</label>
          <input
            type="text"
            id="fname"
            name="fname"
            placeholder="First Name"
            value={firstName}
            onChange={handleFirstNameChange}
          />
        </div>

        <div className="form-row">
          <label htmlFor="lname">Last Name:</label>
          <input
            type="text"
            id="lname"
            name="lname"
            placeholder="Last Name"
            value={lastName}
            onChange={handleLastNameChange}
          />
        </div>
        <div className="form-row">
          <label htmlFor="email">Email:</label>
          <input
            type="email"
            id="email"
            name="email"
            placeholder="Email"
            value={email}
            onChange={handleEmailChange}
          />
        </div>
        <div className="form-row">
          <label htmlFor="password">Password:</label>
          <input
            type="text"
            id="password"
            name="password"
            placeholder="password"
            value={password}
            onChange={handlePasswordChange}
          />
        </div>
        <div className="form-row">
          <label htmlFor="phonenumber">Phone Number:</label>
          <input
            input
            type="tel"
            id="phonenumber"
            name="phonenumber"
            placeholder="Phone Number"
            value={phoneNumber}
            onChange={handlePhoneNumberChange}
          />
        </div>
        <div className="form-buttons">
          <button type="submit" className="register-btn">
            Register
          </button>
          <button
            type="submit"
            className="login-btn"
            onClick={() => onLoginClick("login")}
          >
            Login
          </button>
        </div>
      </form>
    </div>
  );
}

export default RegisterUser;
