
// Create the Traces
var trace1 = {
  x: data.year,
  y: data.poverty,
  mode: "markers",
  type: "scatter",
  name: "Headcount poverty rate",
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
  name: "GDP growth rate",
  marker: {
    color: "orange",
    symbol: "diamond-x"
  }
};

var trace3 = {
  x: data.year,
  y: data.percapita_gdp,
  mode: "markers",
  type: "scatter",
  name: "Per capita GDP",
  marker: {
    color: "#E74C3C",
    symbol: "cross"
  }
};

// Create the data array for the plot
var data = [trace1, trace2, trace3];

// Define the plot layout
var layout = {
  title: "Does growth in income help reduction in poverty?",
  xaxis: { title: "Year", 
  		   range: ['1990', '2015']
  		  },
  yaxis: {title: "Value", 
   		  range: [0, 40]
  		}
};


		


// Plot the chart to a div tag with id "plot"
Plotly.newPlot("plot", data, layout);
