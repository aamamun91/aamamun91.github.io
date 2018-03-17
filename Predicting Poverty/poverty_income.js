

var poverty_headcount = {
  type: "scatter",
  mode: "lines",
  x: data.year,
  y: data.poverty,
  line: {color: '#17BECF'}
}

var GDP_growth = {
  type: "scatter",
  mode: "lines",
  x: data.year,
  y: data.gdp_growth,
  line: {color: '#7F7F7F'}
}

var data = [poverty_headcount,GDP_growth];

var layout = {
  title: 'GDP per capita vs poverty headcount',
  xaxis: {
    range: ['1990', '2013']
  },
  yaxis: {
    autorange: true,
    range: [0, 50],
    type: 'linear'
  }
};

Plotly.newPlot('myDiv', data, layout);