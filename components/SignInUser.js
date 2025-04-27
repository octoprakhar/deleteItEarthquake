import { auth } from "../firebase";
import { db } from "../firebase";
import { doc, getDoc } from "firebase/firestore";
import { signInWithEmailAndPassword } from "firebase/auth";
import { useState } from "react";

function LoginUser({ onRegisterClick, setPosition }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleEmailChange = (event) => {
    setEmail(event.target.value);
  };

  const handlePasswordChange = (event) => {
    setPassword(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const userCredential = await signInWithEmailAndPassword(
        auth,
        email,
        password
      );
      const user = userCredential.user;

      console.log("User signed in:", user);

      //  Fetch user location from Firestore
      const userDocRef = doc(db, "users", user.uid);
      const userSnap = await getDoc(userDocRef);

      if (userSnap.exists()) {
        const userData = userSnap.data();
        const { latitude, longitude } = userData;
        setPosition({ lat: latitude, lng: longitude });
      }

      // Clear form after login
      setEmail("");
      setPassword("");
    } catch (error) {
      console.error("Login failed:", error.message);
      alert("Invalid credentials or user does not exist.");
    }
  };

  return (
    <div className="register-cont">
      <p className="register-title">Login Here.</p>
      <form className="register-form" onSubmit={handleSubmit}>
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
        <div className="form-buttons">
          <button type="submit" className="register-btn">
            Log In
          </button>
          <button
            type="submit"
            className="login-btn"
            onClick={() => onRegisterClick("register")}
          >
            Don't have an account? Register
          </button>
        </div>
      </form>
    </div>
  );
}

export default LoginUser;
