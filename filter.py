import json


entry = []
with open('Movies_and_TV_5.txt',encoding='utf-8') as feedjson:
    data = feedjson.readlines()
    for lines in data[:16000]:
        entry.append(json.loads(lines))

with open('Movie_reviews.json', 'w') as outfile:
    json.dump(entry, outfile)