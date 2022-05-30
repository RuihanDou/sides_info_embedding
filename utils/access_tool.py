import pickle as pkl

def save_dict(path: str, map: dict):
    with open(path, 'wb') as handle:
        pkl.dump(map, handle, protocol=pkl.HIGHEST_PROTOCOL)

def load_dict(path: str) -> dict:
    map = {}
    with open(path, 'rb') as handle:
        map = pkl.load(handle)
    return map

