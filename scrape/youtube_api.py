import os


def get_service_name():
    return "youtube"


def get_version():
    return "v3"


def get_key():
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/api_key", 'r') as f:
        return f.readline().strip()
