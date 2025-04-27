

import WorldMap from "./WorldMap";

function MapScreen({datapoint, riskLevel}){

    // data we get are as follows 
    // [{lat: , lang: , depth: , magnitude: },{lat: , lang: , depth: , magnitude: },...so on]

    const sampleData1 = [
        {
            region: "Uttarkashi, Uttarakhand",
            dateTime: "Today, 10:42 AM",
            desc: "Depth: 10km",
            magnitude: "4.7"

        },
        {
            region: "Sikkim, Nepal Border",
            dateTime: "Today, 8:15 AM",
            desc: "Depth: 15km",
            magnitude: "4.2"

        },
        {
            region: "Sikkim, Nepal Border",
            dateTime: "Today, 8:15 AM",
            desc: "Depth: 15km",
            magnitude: "4.2"

        },

    ]

    const sampleData2 = [
        {
            region: "Uttarkashi, Uttarakhand",
            dateTime: "Today, 10:42 AM",
            desc: "Affected high amount of area",
            magnitude: "4.7"

        },
        {
            region: "Sikkim, Nepal Border",
            dateTime: "Today, 8:15 AM",
            desc: "Affected high amount of area",
            magnitude: "4.2"

        },
        {
            region: "Sikkim, Nepal Border",
            dateTime: "Today, 8:15 AM",
            desc: "Affected high amount of area",
            magnitude: "4.2"

        },

    ]

    return <div className="map-screen-cont">
        <div className="map-screen-map">
            <WorldMap dataPoints={datapoint}/>
        </div>
        <div className="map-screen-other-details-cont">
            <div className="recent-activity-cont">
                    <MapScreenComponent title={"Recent Activity"} data={sampleData1}/>
            </div>
            <div className="world-historical-event-cont">
                <MapScreenComponent title={"Historical Events"} data={sampleData2}/>
                
            </div>
            <div className="risk-assessment-cont">
                <p className="map-screen-risk-assessment-title">Risk Assessment</p>
                <div className="map-screen-risk-assessment-body">
                    <p> Current Risk Level</p>
                    <p style={{ color: 'red' }}>High</p>
                    <hr/>
                </div>
                <p>Located in seisemic zone 4 of India's seismic zone map.</p>
            </div>

        </div>
    </div>
}


function MapScreenComponent({title, data}){
    return <>
    <p className="map-screen-component-title">{title}</p>
    {
        data.map((item,index) => {
            return <MapScreenComponentItem key={index} region={item.region} dateTime={item.dateTime} desc={item.desc} magnitude={item.magnitude}/>
        })
    }
    </>

}

function MapScreenComponentItem({region, dateTime, desc, magnitude}){
    return <div className="map-screen-component-item">
        <p className="map-screen-component-mag">{magnitude}</p>
        <div className="map-screen-component-item-details">
            <p className="map-screen-component-region">{region}</p>
            <div className="map-screen-other-detail">
                <p className="map-screen-component-date-time">{dateTime} ::</p>
                <p className="map-screen-component-desc">{desc}</p>
                

            </div>

        </div>
        
    </div>




}

export default MapScreen;

