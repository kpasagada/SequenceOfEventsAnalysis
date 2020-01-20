import json

if __name__ == "__main__":

    file_names = ['event_09_2018.json', 'event_10_2018.json', 'event_11_2018.json', 'event_12_2018.json']

    json_array = []

    for name in file_names:
        print("Reading file:", name)
        with open(name, 'r') as file:
            for line in file:
                json_array.append(json.loads(line))
        file.close()

    print("Total file size:", len(json_array))

    print("Writing to combined file..")
    with open('event_combined.json', 'a') as file:
        for row in json_array:
            file.write(json.dumps(row) + "\n")
    file.close()
