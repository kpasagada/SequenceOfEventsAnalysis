import urllib.request
import json

if __name__ == "__main__":

    # API key here
    api_key = "A4LkIg8jj24xOQhqbIeCqUTjTZyEXCaS"
    url = "http://eventdata.utdallas.edu/api/data?api_key="

    # Change month and year
    year = "2018"
    month = "09"

    # Change query here if required
    query = '{"$and":[{"month":"' + month + '"},{"year":"' + year + '"}]}'

    with urllib.request.urlopen(url + api_key + '&query=' + query) as url:
        data = json.loads(url.read().decode())
        print('Number of records: ' + str(len(data['data'])))
        with open('event_' + month + '_' + year + '.json', 'a') as file:
            # Writes the entire data array to file
            for row in data['data']:
                file.write(json.dumps(row) + "\n")
        file.close()

