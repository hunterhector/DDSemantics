import requests
import getpass
from urllib.parse import quote
import json


def call_tagme(in_file, out_file, freebase_map, username, password):
    url = 'https://tagme.d4science.org/tagme/tag'

    params = {
        'lang': 'en',
        'gcube-token': '3ccca27e-d0a1-4752-a830-b906b7c089fa-843339462',
        'text': in_file.read()
    }

    r = requests.post(url, auth=(username, password), data=params)

    with open(out_file, 'w') as out:
        tagme_json = json.loads(r.text)
        for spot in tagme_json['annotations']:
            wiki_title = get_wiki_name(spot['title'])
            fbid = freebase_map.get(wiki_title, '/m/UNK')
            spot['mid'] = fbid
        json.dump(tagme_json, out)
        out.write('\n')


def get_wiki_name(name):
    return name.title().replace(' ', '_')


def main(in_dir, out_dir, freebase_map_file):
    import os
    username = input('Username:')
    password = getpass.getpass('Password:')

    freebase_map = {}
    print("Reading freebase id.")
    with open(freebase_map_file) as infile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                mid, wiki_name = parts
                freebase_map[wiki_name] = mid
    print("Done.")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fn in os.listdir(in_dir):
        if fn.endswith('.txt'):
            with open(os.path.join(in_dir, fn)) as infile:
                out_path = os.path.join(out_dir, fn + '.json')
                call_tagme(infile, out_path, freebase_map, username, password)


if __name__ == '__main__':
    import sys

    main(sys.argv[1], sys.argv[2], sys.argv[3])
