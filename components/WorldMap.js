import React from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import L from "leaflet";

// Custom red dot icon
const redDotIcon = new L.DivIcon({
  className: "custom-red-dot",
  html: '<div style="width: 12px; height: 12px; background-color: red; border-radius: 50%;"></div>',
  iconSize: [12, 12],
  iconAnchor: [6, 6],
});

function WorldMap({ dataPoints }) {
  return (
    <MapContainer
      center={[22.9734, 78.6569]} // Center of India
      zoom={5}
      style={{ height: "100%", width: "100%" }}
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {/* Multiple Earthquake Markers */}
      {dataPoints &&
        dataPoints.map((point, idx) => (
          <Marker key={idx} position={[point.latitude, point.longitude]} icon={redDotIcon}>
            <Popup>
              <div>
                <p><strong>Latitude:</strong> {point.latitude}</p>
                <p><strong>Longitude:</strong> {point.longitude}</p>
                <p><strong>Depth:</strong> {point.depth} km</p>
                <p><strong>Magnitude:</strong> {point.mag}</p>
              </div>
            </Popup>
          </Marker>
        ))}
    </MapContainer>
  );
}

export default WorldMap;
