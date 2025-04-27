import { useEffect, useState } from "react";

// News screen Starts here
function NewsAndUpdatesScreen({ position }) {
  //These values will come from api at the time of mounting of this component
  const [temperature, setTemperature] = useState(null);
  const [windSpeed, setWindSpeed] = useState(null);
  const [feelsLike, setFeelsLike] = useState(null);
  const [humidity, setHumidity] = useState(null);
  const [precipitation, setPrecipitation] = useState(null);
  const [condition, setCondition] = useState(null);

  const { lat, lng } = position;

  useEffect(() => {
    const fetchWeatherData = async () => {
      try {
        const response = await fetch("http://localhost:5000/weatherInfo", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ latitude: lat, longitude: lng }),
        });

        const data = await response.json();

        if (data.code === 1 && data.body) {
          const {
            temperature,
            windSpeed,
            feelsLike,
            humidity,
            precipitation,
            condition,
          } = data.body;

          console.log(`Got Precip: ${precipitation}`);

          setTemperature(temperature);
          setWindSpeed(windSpeed);
          setFeelsLike(feelsLike);
          setHumidity(humidity);
          setPrecipitation(precipitation);
          setCondition(condition);
        }
      } catch (error) {
        console.error("Error fetching weather data:", error);
      }
    };

    fetchWeatherData();
  }, [lat, lng, position]);
  const sampleData1 = [
    {
      img: "weather.jpg",
      title: "Magnitude 4.3 earthquake reported in Uttarakhand region",
      desc: "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. ",
      tag: "Breaking",
      source: "National Disaster response force",
      time: "3 hours ago",
    },
    {
      img: "weather.jpg",
      title: "Magnitude 4.3 earthquake reported in Uttarakhand region",
      desc: "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. ",
      tag: "Breaking",
      source: "National Disaster response force",
      time: "3 hours ago",
    },
    {
      img: "weather.jpg",
      title: "Magnitude 4.3 earthquake reported in Uttarakhand region",
      desc: "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. ",
      tag: "Breaking",
      source: "National Disaster response force",
      time: "3 hours ago",
    },
    {
      img: "weather.jpg",
      title: "Magnitude 4.3 earthquake reported in Uttarakhand region",
      desc: "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. Nulla consequat massa quis enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget, arcu. ",
      tag: "Breaking",
      source: "National Disaster response force",
      time: "3 hours ago",
    },
  ];

  return (
    <div className="news-and-updates-cont">
      <div className="news-cont">
        <svg
          viewBox="-2.4 -2.4 28.80 28.80"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          transform="rotate(-45)"
          height={50}
          width={50}
        >
          <g id="SVGRepo_bgCarrier" stroke-width="0">
            <rect
              x="-2.4"
              y="-2.4"
              width="28.80"
              height="28.80"
              rx="14.4"
              fill="#7ed0ec"
              strokewidth="0"
            ></rect>
          </g>
          <g
            id="SVGRepo_tracerCarrier"
            stroke-linecap="round"
            stroke-linejoin="round"
          ></g>
          <g id="SVGRepo_iconCarrier">
            {" "}
            <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M9.94531 1.25H14.0551C15.4227 1.24998 16.525 1.24996 17.3919 1.36652C18.292 1.48754 19.0499 1.74643 19.6518 2.34835C20.2538 2.95027 20.5126 3.70814 20.6337 4.60825C20.7502 5.47522 20.7502 6.57754 20.7502 7.94513V16.0549C20.7502 17.4225 20.7502 18.5248 20.6337 19.3918C20.5126 20.2919 20.2538 21.0497 19.6518 21.6517C19.0499 22.2536 18.292 22.5125 17.3919 22.6335C16.525 22.75 15.4226 22.75 14.0551 22.75H9.94532C8.57773 22.75 7.4754 22.75 6.60844 22.6335C5.70833 22.5125 4.95045 22.2536 4.34854 21.6517C3.74662 21.0497 3.48773 20.2919 3.36671 19.3918C3.32801 19.1039 3.30216 18.7902 3.2849 18.4494C3.24582 18.326 3.23821 18.1912 3.26895 18.0568C3.25016 17.4649 3.25017 16.7991 3.25019 16.0549V7.94513C3.25017 6.57754 3.25015 5.47522 3.36671 4.60825C3.48773 3.70814 3.74662 2.95027 4.34854 2.34835C4.95045 1.74643 5.70833 1.48754 6.60843 1.36652C7.4754 1.24996 8.57772 1.24998 9.94531 1.25ZM4.77694 18.2491C4.79214 18.6029 4.81597 18.914 4.85333 19.1919C4.95199 19.9257 5.13243 20.3142 5.4092 20.591C5.68596 20.8678 6.07453 21.0482 6.80831 21.1469C7.56366 21.2484 8.56477 21.25 10.0002 21.25H14.0002C15.4356 21.25 16.4367 21.2484 17.1921 21.1469C17.9258 21.0482 18.3144 20.8678 18.5912 20.591C18.7875 20.3947 18.9353 20.1421 19.0399 19.75H8.00019C7.58597 19.75 7.25019 19.4142 7.25019 19C7.25019 18.5858 7.58597 18.25 8.00019 18.25H19.2234C19.2419 17.819 19.2477 17.3246 19.2494 16.75H7.89796C6.91971 16.75 6.5777 16.7564 6.31562 16.8267C5.5963 17.0194 5.02286 17.5541 4.77694 18.2491ZM19.2502 15.25H7.89796C7.85879 15.25 7.8202 15.25 7.78217 15.25C6.9642 15.2497 6.40605 15.2495 5.92739 15.3778C5.49941 15.4925 5.10242 15.6798 4.75019 15.9259V8C4.75019 6.56458 4.75178 5.56347 4.85333 4.80812C4.95199 4.07435 5.13243 3.68577 5.4092 3.40901C5.68596 3.13225 6.07453 2.9518 6.80831 2.85315C7.56366 2.75159 8.56477 2.75 10.0002 2.75H14.0002C15.4356 2.75 16.4367 2.75159 17.1921 2.85315C17.9258 2.9518 18.3144 3.13225 18.5912 3.40901C18.8679 3.68577 19.0484 4.07435 19.147 4.80812C19.2486 5.56347 19.2502 6.56458 19.2502 8V15.25ZM7.25019 7C7.25019 6.58579 7.58597 6.25 8.00019 6.25H16.0002C16.4144 6.25 16.7502 6.58579 16.7502 7C16.7502 7.41421 16.4144 7.75 16.0002 7.75H8.00019C7.58597 7.75 7.25019 7.41421 7.25019 7ZM7.25019 10.5C7.25019 10.0858 7.58597 9.75 8.00019 9.75H13.0002C13.4144 9.75 13.7502 10.0858 13.7502 10.5C13.7502 10.9142 13.4144 11.25 13.0002 11.25H8.00019C7.58597 11.25 7.25019 10.9142 7.25019 10.5Z"
              fill="#1C274D"
            ></path>{" "}
          </g>
        </svg>

        <p className="news-cont-title">Disaster News Updates</p>
        <p className="news-cont-desc">
          Latest news on disasters and emergency situations worldwide.
        </p>
        <div className="news-cont-body">
          {sampleData1.map((item, index) => (
            <NewsContainerComponent key={index} {...item} />
          ))}
        </div>
      </div>
      <div className="weather-cont">
        <p className="weather-cont-title">Weather Forcast</p>
        <p className="weather-cont-desc">
          Current condition and forcast for your location
        </p>
        <div className="today-weather-cont">
          <svg
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            height={10}
            width={10}
          >
            <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
            <g
              id="SVGRepo_tracerCarrier"
              stroke-linecap="round"
              stroke-linejoin="round"
            ></g>
            <g id="SVGRepo_iconCarrier">
              {" "}
              <path
                d="M3 13.6493C3 16.6044 5.41766 19 8.4 19L16.5 19C18.9853 19 21 16.9839 21 14.4969C21 12.6503 19.8893 10.9449 18.3 10.25C18.1317 7.32251 15.684 5 12.6893 5C10.3514 5 8.34694 6.48637 7.5 8.5C4.8 8.9375 3 11.2001 3 13.6493Z"
                stroke="#164ff8"
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
              ></path>{" "}
            </g>
          </svg>
          <div className="today-weather-cont-item">
            <p className="today-weather-degree">
              {temperature ? temperature : 12}&deg;C
            </p>
            <p className="today-weather-desc">
              {condition ? condition : "N/A"}
            </p>
          </div>
        </div>
        <div className="weather-other-details">
          <OtherWeatherDetails
            detail={windSpeed ? `${windSpeed} km/h` : "12 km/h NE"}
            className={"other-weather-detail-item1"}
          >
            <p>💨 Wind</p>
          </OtherWeatherDetails>
          <OtherWeatherDetails
            detail={humidity ? `${humidity} g/m` : "12 km/h NE"}
            className={"other-weather-detail-item2"}
          >
            <p>🥵 Humidity</p>
          </OtherWeatherDetails>
          <OtherWeatherDetails
            detail={`${precipitation} mm`}
            className={"other-weather-detail-item3"}
          >
            <p>💧 Precipitation</p>
          </OtherWeatherDetails>
          <OtherWeatherDetails
            detail={feelsLike ? `${feelsLike} C` : "12 km/h NE"}
            className={"other-weather-detail-item4"}
          >
            <p>😊 Feels Like</p>
          </OtherWeatherDetails>
        </div>

        <p className="weather-forecast-title">3-Days Forcast</p>
        <WeatherForecastItem
          day={"Tomorrow"}
          desc={"Sunny"}
          degreeHigh={"44"}
          degreeLow={"41"}
        />
        <hr />
        <WeatherForecastItem
          day={"day after Tomorrow"}
          desc={"Sunny"}
          degreeHigh={"43"}
          degreeLow={"40"}
        />
        <hr />
        <WeatherForecastItem
          day={"2 days after"}
          desc={"Sunny"}
          degreeHigh={"41"}
          degreeLow={"39"}
        />
        <hr />
      </div>
      <div className="weather-alert-cont">
        <p className="weather-alert-title">Weather Alerts</p>
        <WeatherAlertItem
          title={"Monsoon Alert"}
          desc={
            "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis"
          }
          time={"Next 24hrs"}
        />
        <WeatherAlertItem
          title={"Monsoon Alert"}
          desc={
            "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec quam felis"
          }
          time={"Next 24hrs"}
        />
      </div>
    </div>
  );
}

function OtherWeatherDetails({ children, detail, className }) {
  return (
    <div className={className}>
      {children}
      <p>{detail}</p>
    </div>
  );
}

function WeatherForecastItem({ day, desc, degreeHigh, degreeLow }) {
  return (
    <div className="weather-forecast-item-cont">
      <p>{day}</p>
      <p>{desc}</p>
      <p>
        {degreeHigh}&deg;/{degreeLow}&deg;
      </p>
    </div>
  );
}

function WeatherAlertItem({ title, desc, time }) {
  return (
    <div className="weather-alert-item-cont">
      <p className="weather-alert-item-title">{title}</p>
      <p className="weather-alert-item-desc">{desc}</p>
      <p className="weather-alert-item-time">Effective Until: {time}</p>
    </div>
  );
}

function NewsContainerComponent({ img, title, desc, tag, source, time }) {
  return (
    <div className="news-item-cont">
      <img src={img} className="news-item-img" />
      <div className="news-item-text">
        <div className="news-item-headers">
          <p className="news-item-tag">{tag}</p>
          <p className="news-item-time">{time}</p>
        </div>
        <p className="news-item-title">{title}</p>
        <p className="news-item-desc">{desc}</p>
        <div className="news-item-footer">
          <p className="news-item-source">Source: {source}</p>
          <a href="#" className="news-item-read-more">
            Read More...{" "}
          </a>
        </div>
      </div>
    </div>
  );
}

export default NewsAndUpdatesScreen;
