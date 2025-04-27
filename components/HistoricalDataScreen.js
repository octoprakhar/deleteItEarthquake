import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import axios from "axios";

function HistoricalDataScreen() {
  const [startDate, setStartDate] = useState();
  const [endDate, setEndDate] = useState();
  const [location, setLocation] = useState();
  const [magnitude, setMagnitude] = useState();
  const [activeButton, setActiveButton] = useState("overview");

  const [filteredData, setFilteredData] = useState([]);

  const [impactedArea, setImpactedArea] = useState();
  const [magnitudeWiseEarthquake, setMagnitudeWiseEarthquake] = useState();

  const handleStartDateChange = (event) => {
    setStartDate(event.target.value);
  };

  const handleEndDateChange = (event) => {
    setEndDate(event.target.value);
  };

  const handleLocationChange = (event) => {
    setLocation(event.target.value);
  };

  const handleMagnitudeChange = (event) => {
    setMagnitude(event.target.value);
  };

  const handleFilterSearch = async () => {
    try {
      const response = await axios.post(
        "http://localhost:5000/filterEarthquakes",
        {
          startDate,
          endDate,
          location,
          magnitude,
        }
      );

      setFilteredData(response.data);
      setActiveButton("overview");
    } catch (error) {
      console.error("Error fetching filtered earthquakes", error);
    }

    setStartDate("");
    setEndDate("");
    setLocation("");
    setMagnitude("");
  };

  //For most affected region
  useEffect(() => {
    const fetchImpactArea = async () => {
      try {
        const response = await axios.get(
          "http://localhost:5000/mostAffectedRegions"
        );
        console.log(response.data);
        setImpactedArea(response.data);
      } catch (error) {
        console.log("Error occured while fetching impacted region", error);
      }
    };

    const fetchMagnitudeData = async () => {
      try {
        const response = await axios.get(
          "http://localhost:5000/magnitudeWiseEarthquake"
        );
        console.log(response.data);
        setMagnitudeWiseEarthquake(response.data);
      } catch (error) {
        console.log("Error occured while fetching impacted region", error);
      }
    };

    if (activeButton === "regional analysis") {
      fetchImpactArea();
    } else if (activeButton === "trend & analysis") {
      fetchMagnitudeData();
    }
  }, [activeButton]);

  const handleActiveButton = (newActiveButton) => {
    setActiveButton(newActiveButton);
    console.log("Onclick triggered");
    console.log(activeButton);
  };

  const handleReset = () => {
    setStartDate("");
    setEndDate("");
    setLocation("");
    setMagnitude("");
  };

  return (
    <div className="historical-data-cont">
      <p className="historical-data-title">Historical Earthquake Analysis</p>
      <p className="historical-data-desc">
        Anylyzing patterns and trends of past earthquake events in Indian
        subcontinent.
      </p>

      <div className="historical-data-body">
        <div className="search-section-cont">
          <BodyHeadingComponents
            title={"Filter Historical Data"}
            description={
              "Narrow down your earthquake data by date, location or magnitude."
            }
          />
          <div className="search-section-body">
            <GetOrFetchDetail title={"📅 Start Date"}>
              <input
                type="date"
                placeholder="mm/dd/yyyy"
                className="search-section-field"
                value={startDate}
                onChange={handleStartDateChange}
              />
            </GetOrFetchDetail>
            <GetOrFetchDetail title={"📅 End Date"}>
              <input
                type="date"
                placeholder="mm/dd/yyyy"
                className="search-section-field"
                value={endDate}
                onChange={handleEndDateChange}
              />
            </GetOrFetchDetail>
            <GetOrFetchDetail title={"📍 Location"}>
              <input
                type="text"
                placeholder="eg. Gujarat, Himalayan"
                className="search-section-field"
                value={location}
                onChange={handleLocationChange}
              />
            </GetOrFetchDetail>
            <GetOrFetchDetail title={"🔍 Magnitude"}>
              <input
                type="text"
                placeholder="eg. 5.0"
                className="search-section-field"
                value={magnitude}
                onChange={handleMagnitudeChange}
              />
            </GetOrFetchDetail>
          </div>
          <div className="search-section-btn-cont">
            <CustomButton
              title={"Reset"}
              className="reset-btn"
              onclick={handleReset}
            />
            <CustomButton
              title={"Filter Search"}
              className="filter-search-btn"
              onclick={handleFilterSearch}
            />
          </div>
        </div>

        <div className="search-result-des-btn-cont">
          <CustomButton
            title={"Overview"}
            className={
              activeButton === "overview"
                ? "search-result-des-btn active-btn-hist-scr"
                : "search-result-des-btn"
            }
            onclick={() => handleActiveButton("overview")}
          />
          <CustomButton
            title={"Trend & Analysis"}
            className={
              activeButton === "trend & analysis"
                ? "search-result-des-btn active-btn-hist-scr"
                : "search-result-des-btn"
            }
            onclick={() => handleActiveButton("trend & analysis")}
          />
          <CustomButton
            title={"Regional Analysis"}
            className={
              activeButton === "regional analysis"
                ? "search-result-des-btn active-btn-hist-scr"
                : "search-result-des-btn"
            }
            onclick={() => handleActiveButton("regional analysis")}
          />
          <CustomButton
            title={"Notable Events"}
            className={
              activeButton === "notable events"
                ? "search-result-des-btn active-btn-hist-scr"
                : "search-result-des-btn"
            }
            onclick={() => handleActiveButton("notable events")}
          />
        </div>
        <div className="search-result-cont">
          {activeButton === "overview" ? (
            <OverviewContainer filteredData={filteredData} />
          ) : activeButton === "trend & analysis" ? (
            <TrendAnalysisContainer datapoints={magnitudeWiseEarthquake} />
          ) : activeButton === "notable events" ? (
            <NotableEvent />
          ) : (
            <RegionalAnalysis datapoints={impactedArea} />
          )}
        </div>
      </div>
    </div>
  );
}

function OverviewContainer({ filteredData }) {
  const totalEvents = filteredData.length;

  const avgMagnitude = totalEvents
    ? (
        filteredData.reduce((sum, quake) => sum + (quake.mag || 0), 0) /
        totalEvents
      ).toFixed(2)
    : 0;

  const significantEvents = filteredData.filter(
    (quake) => quake.mag >= 6
  ).length;
  const tsunamiEvents = filteredData.filter(
    (quake) => quake.tsunami === 1
  ).length;

  return (
    <>
      <BodyHeadingComponents
        title={"Earthquake Data Summary"}
        description={
          "Data from 1990-2023 for the Indian subcontinents and surrounding regions"
        }
      />
      <div className="search-result-body">
        <GetOrFetchDetail title={"Total Events"}>
          <p className="search-result-overview-detail">{totalEvents}</p>
        </GetOrFetchDetail>
        <GetOrFetchDetail title={"Average Magnitude"}>
          <p className="search-result-overview-detail">{avgMagnitude}</p>
        </GetOrFetchDetail>
        <GetOrFetchDetail title={"Significant Events"}>
          <p className="search-result-overview-detail">{significantEvents}</p>
        </GetOrFetchDetail>
        <GetOrFetchDetail title={"Tsunami Events"}>
          <p className="search-result-overview-detail">{tsunamiEvents}</p>
        </GetOrFetchDetail>
      </div>
    </>
  );
}

function TrendAnalysisContainer({ datapoints }) {
  if (!datapoints || !Array.isArray(datapoints)) {
    return <p>Loading or no data available...</p>;
  }

  const chartData = datapoints.map((obj) => {
    const [mag, freq] = Object.entries(obj)[0];
    return {
      magnitude: mag,
      frequency: freq,
    };
  });
  return (
    <>
      <BodyHeadingComponents
        title={"Magnitude Distribution"}
        description={"Frequency of earthquake by magnitude range"}
      />
      <div
        className="search-result-body"
        style={{ width: "100%", height: 400 }}
      >
        {/* To make graph here by datapoints as {mag:freq,mag:freq} */}
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="magnitude"
              label={{
                value: "Magnitude",
                position: "insideBottom",
                offset: -5,
              }}
            />
            <YAxis
              label={{ value: "Frequency", angle: -90, position: "insideLeft" }}
            />
            <Tooltip />
            <Bar dataKey="frequency" fill="#8884d8" radius={[8, 8, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}

function NotableEvent() {
  //Get this data from server
  const data = [
    {
      region: "Nepal-India Border",
      date: "November 9, 2022",
      desc: "Minor damage in western Nepal and Uttarakhand",
      magnitude: "2.1",
      severity: "mild",
      depth: "15 km",
    },
    {
      region: "Manipur",
      date: "January 4, 2016",
      desc: "Moderate damage to building in Imphal",
      magnitude: "6.7",
      severity: "moderate",
      depth: "55 km",
    },
    {
      region: "Nepal(affecting North India)",
      date: "April 25, 2015",
      desc: "Widespread destruction, affecting Bihar and Uttar Pradesh in India",
      magnitude: "7.8",
      severity: "severe",
      depth: "8.2 km",
    },
  ];

  return (
    <>
      <BodyHeadingComponents
        title={"Recent Major Events"}
        description={"Selected High impact earthquakes in recent years."}
      />
      <div className="search-result-notable-event-cont">
        {data.map((item, index) => (
          <NotableEventItem key={index} {...item} />
        ))}
      </div>
    </>
  );
}

function interpolateColor(color1, color2, factor) {
  const result = color1.slice();
  for (let i = 0; i < 3; i++) {
    result[i] = Math.round(result[i] + factor * (color2[i] - color1[i]));
  }
  return result;
}

function rgbToHex(rgb) {
  return (
    "#" +
    rgb
      .map((val) => {
        const hex = val.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
  );
}

function getGradientColor(index, total) {
  const percent = index / (total - 1);

  const maroon = [128, 0, 0];
  const red = [255, 0, 0];
  const yellow = [255, 255, 0];

  let interpolatedColor;

  if (percent <= 0.5) {
    const localPercent = percent / 0.5;
    interpolatedColor = interpolateColor(maroon, red, localPercent);
  } else {
    const localPercent = (percent - 0.5) / 0.5;
    interpolatedColor = interpolateColor(red, yellow, localPercent);
  }

  return rgbToHex(interpolatedColor);
}

function RegionalAnalysis({ datapoints }) {
  if (!datapoints || !Array.isArray(datapoints)) {
    return <p>Loading or no data available...</p>;
  }
  // Reformat the data
  const formattedData = datapoints.map((obj) => {
    const region = Object.keys(obj)[0];
    return { region, count: obj[region] };
  });

  const sortedData = [...formattedData].sort((a, b) => b.count - a.count);

  return (
    <>
      <BodyHeadingComponents
        title="Regional Earthquake Impact"
        description="Most earthquake impacted regions of the world"
      />
      <div
        className="search-result-body"
        style={{ width: "100%", height: 500 }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={sortedData}
            margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
          >
            <XAxis type="number" />
            <YAxis dataKey="region" type="category" width={160} />
            <Tooltip />
            <Bar dataKey="count">
              {sortedData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={getGradientColor(index, sortedData.length)}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}

// Other smaller components used in this screen

function NotableEventItem({ region, date, desc, magnitude, severity, depth }) {
  return (
    <div className="notable-event-item-cont">
      <p className="notable-event-mag-sec">{magnitude}</p>
      <div className="notable-event-regiion-detail">
        <p className="notable-event-region">{region}</p>
        <p className="notable-event-date">{date}</p>
        <p className="notable-event-desc">{desc}</p>
      </div>
      <div className="notable-event-other-detail">
        <p
          className={
            severity === "mild"
              ? "mild-severity"
              : severity === "moderate"
              ? "moderate-severity"
              : "severe-severity"
          }
        >
          {severity}
        </p>
        <p className="notable-event-depth">{depth}</p>
      </div>
    </div>
  );
}
function BodyHeadingComponents({ title, description }) {
  return (
    <div className="body-heading-cont">
      <p className="body-heading-title">{title}</p>
      <p className="body-heading-desc">{description}</p>
    </div>
  );
}

function GetOrFetchDetail({ title, children }) {
  return (
    <div className="get-or-fetch-detail-cont">
      <p className="search-section-title">{title}</p>
      {children}
    </div>
  );
}

function CustomButton({ title, className, onclick = () => {} }) {
  return (
    <button className={className} onClick={onclick}>
      {title}
    </button>
  );
}

export default HistoricalDataScreen;
