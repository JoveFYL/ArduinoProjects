import requests

baseurl = 'https://rickandmortyapi.com/api/'
endpoint = 'character'
r = requests.get(baseurl + endpoint)
data = r.json()

for item in data['results']:
    print(item['name'], len(item['episode']))