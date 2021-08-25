import json


def get_config():
    with open('config.json') as json_file:
        return json.load(json_file)