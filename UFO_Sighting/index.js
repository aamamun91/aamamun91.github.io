
var $tbody = document.querySelector("tbody");
var $loadMoreBtn = document.querySelector("#load-btn");
var $dateInput = document.querySelector("#date");
var $cityInput = document.querySelector("#city");
var $stateInput = document.querySelector("#state");
var $countryInput = document.querySelector("#country");
var $shapeInput = document.querySelector("#shape");
var $searchBtn = document.querySelector("#search");
var $loadMoreResults = document.querySelector("#loadsearch-btn");


var startingIndex = 0;
var resultsPerPage = 150;

//var filterUfoData = dataSet;

function renderTable(){


  var endingIndex = startingIndex + resultsPerPage;

  $tbody.innerHTML ="";

  var ufoSubset = dataSet.slice(startingIndex, endingIndex);

  for (var i=0; i< ufoSubset.length; i++){

    var ufo = ufoSubset[i];

    var fields = Object.keys(ufo);
    var $row = $tbody.insertRow(i);

    for (var j=0; j< fields.length; j++){

      var field = fields[j];
      var $cell = $row.insertCell(j);

      $cell.innerText = ufo[field];

    }
  }
}

$loadMoreBtn.addEventListener("click", handleButtonClick);

function handleButtonClick() {
  // Increase startingIndex by 100 and render the next section of the table
  startingIndex += resultsPerPage;
  renderTable();
  // Check to see if there are any more results to render
  if (startingIndex + resultsPerPage >= dataSet.length) {
    $loadMoreBtn.classList.add("disabled");
    $loadMoreBtn.innerText = "All UFO Loaded";
    $loadMoreBtn.removeEventListener("click", handleButtonClick);
  }
}

var startingIndex = 0;
var resultsPerPage = 50;

$searchBtn.addEventListener("click", handleSearchButtonClick); 

function handleSearchButtonClick(){

  var filterDate = $dateInput.value;
  var filterCity = $cityInput.value.trim().toLowerCase();
  var filterState = $stateInput.value.trim().toLowerCase();
  var filterCountry = $countryInput.value.trim().toLowerCase();
  var filterShape = $shapeInput.value.trim().toLowerCase();


  filterUfoData = dataSet.filter(function(ufo){

    var ufodate = ufo.datetime;
    var ufocity = ufo.city.substring(0, filterCity.length).toLowerCase();
    var ufostate = ufo.state.substring(0, filterState.length).toLowerCase();
    var ufocountry = ufo.country.substring(0, filterCountry.length).toLowerCase();
    var ufoshape = ufo.shape.substring(0, filterShape.length).toLowerCase();

  //   if (startingIndex + resultsPerPage >= dataSet.length) {
  //   $loadMoreResults.classList.add("disabled");
  //   $loadMoreResults.innerText = "All Results Loaded";
  //   $loadMoreResults.removeEventListener("click", handleSearchButtonClick);
  // }

    if ((ufodate === filterDate) && (ufocity === filterCity) && (ufostate === filterState) && (ufocountry === filterCountry) && (ufoshape === filterShape)) {
      return true;
    }
    return false;    
    
  });

  renderTable();
}

renderTable();

