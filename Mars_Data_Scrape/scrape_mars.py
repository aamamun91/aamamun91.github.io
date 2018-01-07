

# Dependencies
from bs4 import BeautifulSoup
import requests
import pymongo
import time
from splinter import Browser
import pandas as pd


def scrape():

	marsDict={}
	executable_path = {'executable_path': 'chromedriver'}
	browser = Browser('chrome', **executable_path, headless=True)

	nasa_url = 'https://mars.nasa.gov/news/'
	browser.visit(nasa_url)
	time.sleep(1)
	html = browser.html

	# Create BeautifulSoup object; parse with 'html.parser'
	nasa_soup = BeautifulSoup(html, 'html.parser')
	news_title = nasa_soup.find("div", class_="content_title").get_text()
	news_p = nasa_soup.find("div", class_="rollover_description").get_text()

	marsDict['news_title'] = news_title
	marsDict['news_teaser'] = news_p
	print("News scraped")



	jpl_url = 'https://www.jpl.nasa.gov/spaceimages/?search=&category=Mars'
	baseUrl = 'https://www.jpl.nasa.gov'
	browser.visit(jpl_url)
	browser.click_link_by_partial_text('FULL IMAGE')
	jplhtml = browser.html

	jpl_soup = BeautifulSoup(jplhtml, 'html.parser')
	more_info = jpl_soup.find('a', class_='button fancybox').get('data-link')
	more_info = baseUrl + more_info
	browser.visit(more_info)
	moreinfohtml = browser.html
	moreinfosoup = BeautifulSoup(moreinfohtml, 'html.parser')
	figure = moreinfosoup.find('figure', class_='lede')
	featured_image_url = figure.find('a').get('href')
	featured_image_url = baseUrl + featured_image_url
	marsDict['featured_image_url'] = featured_image_url
	print("Featured image scraped")



	mars_weather_url ='https://twitter.com/marswxreport?lang=en'
	browser.visit(mars_weather_url)
	mars_weather_html = browser.html
	mars_weather_soup = BeautifulSoup(mars_weather_html, 'html.parser')
	mars_weather_tweet = mars_weather_soup.find("p", class_="TweetTextSize TweetTextSize--normal js-tweet-text tweet-text").get_text()
	mars_weather = mars_weather_tweet

	marsDict['mars_weather'] = mars_weather_tweet
	print("Weather scraped")


	imageurl = 'https://space-facts.com/mars/'
	browser.visit(imageurl)
	soup = BeautifulSoup(browser.html,'html5lib')
	table = soup.find('table',class_="tablepress tablepress-id-mars")
	df = pd.read_html(str(table))
	tableHTML = df[0].to_html(index=False, escape=True, header=None)

	htmlTable = tableHTML.replace('\n', '')
	marsDict['factTable'] = htmlTable
	print("Fact table scraped")


	hemispheresurl = 'https://astrogeology.usgs.gov/search/results?q=hemisphere+enhanced&k1=target&v1=Mars'
	hemisphereBaseUrl = 'https://astrogeology.usgs.gov'
	browser.visit(hemispheresurl)
	soup = BeautifulSoup(browser.html,'html5lib')
	hemispheres = soup.find('div', class_='collapsible results').find_all('a')
	hemisphere_image_urls = []
	hemispheredict = {}

	for hemisphere in hemispheres:
		hemisphereLink = hemisphere.get('href')
		browser.visit(hemisphereBaseUrl + hemisphereLink)
		soup = BeautifulSoup(browser.html, 'html.parser')
		title = soup.find('title').text
		hemisphereTitle = title.split('|')
		hemisphereTitle = hemisphereTitle[0].replace(' Enhanced ','')
		imgUrl = soup.find('img',class_='wide-image').get('src')
		imgUrl = hemisphereBaseUrl + imgUrl
		hemispheredict = {"title": hemisphereTitle, "img_url":imgUrl}
		hemisphere_image_urls.append(hemispheredict)

	marsDict['HemisphereImages'] = hemisphere_image_urls
	print("Hemispheres scraped")
	print(marsDict)

	return(marsDict)
