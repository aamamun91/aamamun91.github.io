

```python
from citipy import citipy as citi
import requests as req
import matplotlib.pyplot as plt
import os 
import csv
import pandas as pd
import numpy as np
import json
```


```python
citi_geocode = []
```


```python
file_path = os.path.join('../WeatherPy/data/worldcities.csv')
```


```python
with open(file_path, newline='') as city_file:
    cities=csv.reader(city_file, delimiter=',')
    #next(cities)
    
    for city in cities:
        geocode = (city[2], city[3])
        citi_geocode.append(geocode)
```


```python
geocode_df = pd.DataFrame(citi_geocode[1:],columns=citi_geocode[0])
```


```python
geocode_sample = geocode_df.sample(500)
```


```python
geocode_sample.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32831</th>
      <td>46.4</td>
      <td>24.266667</td>
    </tr>
    <tr>
      <th>42969</th>
      <td>26.2825000</td>
      <td>-80.1072222</td>
    </tr>
    <tr>
      <th>8713</th>
      <td>50.733333</td>
      <td>8.283333</td>
    </tr>
    <tr>
      <th>8645</th>
      <td>49.397222</td>
      <td>8.533611</td>
    </tr>
    <tr>
      <th>18080</th>
      <td>25.633333</td>
      <td>88.316667</td>
    </tr>
  </tbody>
</table>
</div>




```python
city_df = pd.DataFrame()
city_df ['City Name'] = ""
city_df['Latitude'] = ""
```


```python
for index, row in geocode_sample.iterrows():
    #print(row['Latitude'], row['Longitude'])
    city =  citi.nearest_city(float(row['Latitude']), float(row['Longitude']))
    city_name = city.city_name
    city_df.set_value(index, 'City Name', city_name)
    city_df.set_value(index, 'Latitude', row['Latitude'])
```


```python
city_df.shape
```




    (500, 2)




```python
weather = pd.DataFrame()
weather['City Name'] = ""
weather['Latitude'] =""
weather['Temperature'] =""
weather['Humidity'] =""
weather['Cloudiness'] =""
weather['Wind Speed'] =""
```


```python
api_key = "7be7dfb02672af49d73e9ca689405225"
url = "http://api.openweathermap.org/data/2.5/weather?"
units = "imperial"
```


```python
for index, row in city_df.iterrows():
    # print(row['City Name'])
    try:
        query_url = url + "appid="+api_key + "&units=" + units + "&q="+ row['City Name'] + "&units=" + units
        print(query_url)
        response = req.get(query_url)
        data = response.json()
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        cloudiness = data["clouds"]["all"]
        wind_speed = data["wind"]["speed"]
        weather.set_value(index, 'City Name', row['City Name'])
        weather.set_value(index, 'Latitude', row['Latitude'])
        weather.set_value(index, 'Temperature', temperature)
        weather.set_value(index, 'Humidity', humidity)
        weather.set_value(index, 'Cloudiness', cloudiness)
        weather.set_value(index, 'Wind Speed', wind_speed)
    except:
        continue
```

    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cucerdea&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pompano beach highlands&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dillenburg&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bruhl&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kaliyaganj&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=njinikom&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=chirkey&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=chai badan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=blagoveshchensk&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pyatnitskoye&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ubrique&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bagn&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=villavicencio&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=verkhniy landekh&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=velden am worthersee&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=culaman&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=roshal&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pau brasil&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=barbosa ferraz&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=volos&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=babakan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=samatau&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=rio branco do sul&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=martaban&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=nizhniy bestyakh&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=de panne&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=montemor-o-novo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=wiarton&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=banki&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ocland&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=astara&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=llanera&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=boulogne-billancourt&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=hartford&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=majalengka&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=arambagh&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mosna&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=beresti-bistrita&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=caleta de carquin&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pyatigorskiy&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=springville&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=isla&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=waiuku&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bitanhuan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kadipur&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=malappuram&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=zaragoza&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=anahuac&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lewisburg&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=singen&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pully&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bailieborough&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=silanga&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=davenda&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=casa quemada&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=schwetzingen&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=arruda dos vinhos&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=primero de marzo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sorgun&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=wakefield&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=atessa&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=agliana&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=melilli&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=monino&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kipalbig&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bytkiv&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ancud&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ouidah&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=borsodnadasd&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=namyslow&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lukovskaya&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dapdap&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=boromlya&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ivrea&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lagunillas&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=shelui&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=champasak&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dzerzhinsk&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=manpur&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=paramount&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dmitriyevka&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=varnamo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=karari&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=palos verdes estates&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=micomeseng&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=nuristan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=moroni&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=gurjaani&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=arbazh&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ramsgate&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=landang&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=gumaus&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=uberlandia&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=atyuryevo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sulina&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kefalovrison&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=novallas&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=odiong&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mardan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=saratovskaya&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=laureles&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=vladivostok&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=chapleau&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cizre&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=talang&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=tiszasuly&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=amreli&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=alibagu&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=aksaray&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=orsha&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lindenberg&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=krestena&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lingasad&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=tosya&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=chandler&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=paskov&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=popesti&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=basiao&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=coracora&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=newberry&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=aiyinion&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=hamada&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cerro cama&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lulea&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bartlett&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=that phanom&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=staryy cherek&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kohtla-jarve&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bili oslavy&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=shanting&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=san juancito&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mons&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=estancia velha&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=la rinconada&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=jizan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=the village&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=talitsa&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=gadag&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=chingola&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=baktaloranthaza&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=becerril&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=perisani&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ilkal&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=engel-yurt&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sevan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sergio osmena sr&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mudbidri&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=crauthem&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=goragorskiy&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=hickory hills&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kriva palanka&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=macaiba&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=vaitele&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=montrouge&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=havlickuv brod&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=yetkul&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=san narciso&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=burgeo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bascov&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=san antonio de la cal&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=allentown&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=peciu nou&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kalkar&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dumfries&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bagepalli&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=eidfjord&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=paine&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=jefferson&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=luna&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=enfield&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=oakland park&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=temanggung&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bondoukou&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=rombas&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=tepatitlan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=moreira de conegos&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=jacareacanga&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=serramanna&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cartago&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bucerias&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=binga&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=luau&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ikskile&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=chanal&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=karachev&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kabrai&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=alabino&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=nova varos&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=tamuin&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sulucan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=alba iulia&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=carmo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=corbeni&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=brinkovskaya&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lohmar&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=yuzhnyy-kospashskiy&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=helsingborg&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pakxe&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=puyallup&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cristesti&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=las margaritas&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=progreso&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bradu&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=jaragua&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=calw&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=quinabigan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=khobi&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=jalu&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mogpog&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=atescatempa&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=catazaja&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=handwara&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=saran&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=troon&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=saint-etienne&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=motozintla&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=stelle&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=anar darreh&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=quattro castella&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sendriceni&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=terracina&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kardamas&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=esteli&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bolshiye berezniki&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=gavar&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mandawar&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sabang&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kato kamila&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=francistown&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=surinam&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=erikoussa&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=margarita&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=smila&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cubulero&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=villa rica&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=paicandu&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=saray&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=parincea&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=stratos&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lady lake&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=purwa&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mairi&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=aban&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=alhandra&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=concord&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=hauge&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pinotepa de don luis&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=manibaug&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=swedru&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=tarrafal&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=balyaga&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=maidenhead&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=araua&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=toyoshina&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=yashan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=krasnyye chetai&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=warren&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=temoaya&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cocal&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pec&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=skali&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=odiongan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=talitsy&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=biri&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=schalksmuhle&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=solin&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=atotonilco el alto&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=green&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dologon&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ozubulu&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=catbalogan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=itoman&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=calasgasan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=san giustino&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=colchester&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=greenfield&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=willimantic&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=khani&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=knezica&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=augustow&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=benton&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=heerlen&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=puerto castilla&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pato-o&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=winchester&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kozlovice&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=brcko&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=carstairs&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=burgdorf&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=abramut&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kopayhorod&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=aptos&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dakor&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sutton in ashfield&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=terdal&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ypsilanti&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cociuba mare&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=concepcion batres&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=potrerillos&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=schertz&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=san miguel&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sao joao da ponte&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=woodbridge&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=daneasa&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=woodend&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=nueva loja&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mvuma&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=periam&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=porlamar&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lufilufi&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=feroke&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kosikha&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=vokhtoga&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=chengdu&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=north creek&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lamont&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=calsib&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pance&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=domanpot&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=iles&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pandamatenga&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=eminabad&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=giroc&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=hendijan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=goderich&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=college&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=protivin&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=policka&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=oildale&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=rongu&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=hatsukaichi&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=chelyabinsk-70&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=labelle&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=concepcion&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dingzhou&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=weilburg&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=harsum&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dos hermanas&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=apulia&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ikere&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=malindi&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=baraya&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ballingry&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=vigo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=hangzhou&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sosnovka&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sale&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=san francisco el alto&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=villa corzo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=santo nino&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lahnstein&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=los palacios&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=nanca&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pingyi&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=raeren&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=jicaro galan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=backo petrovo selo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kosum phisai&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=manta&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=belle fourche&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mangrul pir&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ondo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lebanon&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=korycany&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=miaga&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=paralia&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pirayu&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=madulari&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=palo alto&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sikonge&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=markkleeberg&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=nyamati&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lakewood&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=oud-beijerland&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=old saybrook&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=stoenesti&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=pachalum&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=muragachha&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=urman&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ayia varvara&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=goaigoaza&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=galion&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kefar weradim&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ellisville&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=diadi&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=benevides&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=volodarka&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ridgetown&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=finale emilia&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=laguilayan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=rudnik&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=graaff-reinet&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bridgeton&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ardassa&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=duderstadt&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=valencia&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=santa cruz del valle&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=governador valadares&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=clark&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lake shore&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=murmansk&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=shamkhal&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=nanma&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=loxstedt&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=boca del rio&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=uren&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=waidhofen&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=stabat&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=north andover&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=stirion&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=tank&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=buena vista&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=yelizovo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=murzzuschlag&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=fortaleza&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=tinubdan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cansilayan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=khvastovichi&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=alfonsine&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=puelay&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bron&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=tomelloso&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=atascocita&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=venlo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=doicesti&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=anqing&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=alaca&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ouro preto&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=becsehely&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kesteren&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=leimen&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ocos&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=parede&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=nyborg&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kandhla&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=trudovoye&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=asosa&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=new guinlo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=carmen&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sadsalan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=borup&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lawton&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dvorichna&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=cuxhaven&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=linay&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=vomp&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=santa cruz mulua&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=krasnokholm&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=namikupa&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=xicohtzinco&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=hienghene&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=zhukovo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=huaiyuan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=altos&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=papagaios&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lufkin&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=san cristobal&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=san rafael&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mamaku&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ararat&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=vigia del fuerte&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bweyogerere&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sibiti&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=fairview park&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=herent&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=muttupet&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=rheinberg&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=donsol&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=san felipe&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=seregno&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=langreo&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ashland&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dagshai&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=stroitel&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dessalines&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=yanan&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=poros&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=belebey&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=conceicao do coite&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=budhgaon&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bad krozingen&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=amboise&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=zhelnino&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=silawit&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=bytosh&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=lenti&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=hjelset&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=idappadi&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=altotonga&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=chipiona&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=rutul&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kalamata&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=droitwich&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=mitu&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=morbegno&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kasuga&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=ferrandina&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=dakoro&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=sanford&units=imperial
    http://api.openweathermap.org/data/2.5/weather?appid=7be7dfb02672af49d73e9ca689405225&units=imperial&q=kabansk&units=imperial
    


```python
weather.reset_index(inplace=True)
weather.shape
```




    (500, 7)




```python
weather_df = weather.iloc[:, 1:7]
weather_df.to_csv('output/city_weather.csv')
weather_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>City Name</th>
      <th>Latitude</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Cloudiness</th>
      <th>Wind Speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cucerdea</td>
      <td>46.4</td>
      <td>60.85</td>
      <td>67</td>
      <td>40</td>
      <td>3.36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pompano beach highlands</td>
      <td>26.2825000</td>
      <td>71.37</td>
      <td>60</td>
      <td>1</td>
      <td>9.17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dillenburg</td>
      <td>50.733333</td>
      <td>48.2</td>
      <td>81</td>
      <td>75</td>
      <td>13.87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bruhl</td>
      <td>49.397222</td>
      <td>56.17</td>
      <td>66</td>
      <td>40</td>
      <td>11.41</td>
    </tr>
    <tr>
      <th>4</th>
      <td>kaliyaganj</td>
      <td>25.633333</td>
      <td>72.1</td>
      <td>79</td>
      <td>0</td>
      <td>2.82</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)
ax.scatter('Latitude', 'Temperature', data = weather_df, alpha = 0.7)
ax.set_title("Temperature (F) vs Latitude (10/27/2017)", weight='bold', fontsize=15)
ax.set_xlabel("Latitude", weight='bold', fontsize = 12)
ax.set_ylabel('Temperature (F)', weight ='bold', fontsize=12)
#ax.set_aspect('auto')
fig.tight_layout()
fig.savefig('output/temperatureVsLat.png')
plt.grid()
plt.show()
```


![png](output_15_0.png)



```python
""" The figure above plots temperature measured in fahrenheit against latitude for five hundred cities selected randomly
around the world. From the graph above we find that there is an inverse relationship between latitude and temperature 
where regions at lower latitudes have higher temperatures compared to areas at higher latitudes. For example, city with 
latitude around -20 have the highest temperature of 97 degree fahrenheit whereas city with latitude 60 are having temperature 
of 19 degree fahrenheit. 
"""
```




    ' The figure above plots temperature measured in farenheit against latitude \n\n'




```python
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)
ax.scatter('Latitude', 'Humidity', data = weather_df, alpha = 0.6, color='r')
ax.set_title("Humidity vs Latitude (10/27/2017)", weight='bold', fontsize=15)
ax.set_xlabel("Latitude", weight='bold', fontsize = 12)
ax.set_ylabel('Humidity (%)', weight ='bold', fontsize=12)
fig.tight_layout()
fig.savefig('output/humidityVsLat.png')
plt.grid()
plt.show()
```


![png](output_17_0.png)



```python
"""The figure above plots humidity (%), the amount of water vapor in the air, against latitude for the same set of cities. 
The graph above displays mixed pattern as the city with both higher and lower latitude are having humidity of low and high. 
The cities with latitude 40 have humidity ranging from 25 to 85 percent. With higher latitude the range in humidity is 
40 to 100 percent. Googling about the relationship between humidity and latitude suggests that humidity also depends on the 
regions where the cities are located. """
```


```python
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)
ax.scatter('Latitude', 'Cloudiness', data = weather_df, alpha = 0.6, color='g')
ax.set_title("Cloudiness vs Latitude (10/27/2017)", weight='bold', fontsize=15)
ax.set_xlabel("Latitude", weight='bold', fontsize = 12)
ax.set_ylabel('Cloudiness (%)', weight ='bold', fontsize=12)
fig.tight_layout()
fig.savefig('output/cloudinessVsLat.png')
plt.grid()
plt.show()
```


![png](output_19_0.png)



```python
""" The plot of cloudiness vs latitude for the cities also doesn't reveal any trend. Percent of cloudiness have both high and low 
across latitude for the cities. """
```


```python
fig, ax = plt.subplots()
fig.set_size_inches(10, 7)
ax.scatter('Latitude', 'Wind Speed', data = weather_df, alpha = 0.6, color='y')
ax.set_title("Wind Speed (mph) vs Latitude (10/27/2017)", weight='bold', fontsize=15)
ax.set_xlabel("Latitude", weight='bold', fontsize = 12)
ax.set_ylabel('Wind speed (mph)', weight ='bold', fontsize=12)
fig.tight_layout()
fig.savefig('output/windSpeedVsLat.png')
plt.grid()
plt.show()
```


![png](output_21_0.png)



```python
""" The plot of wind speed measured by miles per hour vs latitude reveal that in most cities selected randomly in this analysis
have low wind speed. However we can't draw any definitive conclusion about relatshiop between winde speed and latitude. 
"""
```
