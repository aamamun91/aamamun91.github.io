
// Create the Traces
var trace1 = {
  x: data.year,
  y: data.poverty,
  mode: "markers",
  type: "scatter",
  name: "Poverty rate (%)",
  marker: {
    color: "#2077b4",
    symbol: "hexagram"
  }
};

var trace2 = {
  x: data.year,
  y: data.gdp_growth,
  mode: "markers",
  type: "scatter",
  name: "GDP growth rate (%)",
  marker: {
    color: "orange",
    symbol: "diamond-x"
  }
};


// Create the data array for the plot
var data = [trace1, trace2];

// Define the plot layout
var layout = {
  title: "How does GDP growth rate helps poverty reduction?",
  xaxis: { title: "Year", 
  		   range: ['1990', '2015']
  		  },
  yaxis: {title: "Value", 
   		  range: [0, 40]
  		}
};
	


// Plot the chart to a div tag with id "plot"
Plotly.newPlot("plot2", data, layout);
