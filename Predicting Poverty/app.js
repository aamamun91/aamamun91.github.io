var svgWidth = 960;
var svgHeight = 500;

var margin = { top: 20, right: 40, bottom: 60, left: 50 };

var width = svgWidth - margin.left - margin.right;
var height = svgHeight - margin.top - margin.bottom;

// Create an SVG wrapper, append an SVG group that will hold our chart, and shift the latter by left and top margins.
var svg = d3
  .select("body")
  .append("svg")
  .attr("width", svgWidth)
  .attr("height", svgHeight)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Import data from the mojoData.csv file

console.log(data)


d3.csv("C:\Users\AMAMUN/Documents/MeasuringPoverty/data/poverty_income.csv", function(error, povertydata) {
  if (error) throw error;

  // Create a function to parse date and time
  var parseTime = d3.timeParse("%d-%b");

  // Format the data
  povertydata.forEach(function(data) {
    data.year = data.year;
    data.poverty = +data.poverty;
    data.gdp_growth = +data.gdp_growth;
  });

  // Set the ranges with scaling functions
  var xTimeScale = d3.scaleTime().range([0, width]);

  var yLinearScale = d3.scaleLinear().range([height, 0]);

  // Functions to create axes
  var bottomAxis = d3.axisBottom(xTimeScale);
  var leftAxis = d3.axisLeft(yLinearScale);

  // Step 1: Set up the x-axis and y-axis domains
  //= =============================================
  var povertyMax = d3.max(povertydata, function(data) {
    return data.poverty;
  });
  var growthMax = d3.max(povertydata, function(data) {
    return data.gdp_growth;
  });

  var yMax = povertyMax > growthMax ? povertyMax : growthMax;

  // Scale the domain
  xTimeScale.domain(
    d3.extent(povertydata, function(data) {
      return data.year;
    })
  );

  // Use the yMax value to set the yLinearScale domain
  yLinearScale.domain([0, yMax]);

  
  // Step 2: Set up two line generators and append two SVG paths
  //= =============================================
  // Line generators for each line
  var line1 = d3
    .line()
    .x(function(data) {
      return xTimeScale(data.year);
    })
    .y(function(data) {
      return yLinearScale(data.poverty);
    });

  var line2 = d3
    .line()
    .x(function(data) {
      return xTimeScale(data.year);
    })
    .y(function(data) {
      return yLinearScale(data.gdp_growth);
    });

  // Append a path for line1
  svg
    .append("path")
    .data([povertydata])
    .attr("d", line1)
    .attr("class", "line green");

  // Append a path for line2
  svg
    .append("path")
    .data([povertydata])
    .attr("d", line2)
    .attr("class", "line orange");

  // Add x-axis
  svg
    .append("g")
    .attr("transform", "translate(0, " + height + ")")
    .call(bottomAxis);

  // Add y-axis
  svg.append("g").call(leftAxis);

 
});
