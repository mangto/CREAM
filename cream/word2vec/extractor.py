import pickle

def extract(original:str, export:str) -> None:
    data = pickle.load(open(original, 'rb'))
    model   = data['model']

    pickle.dump(model, open(export, 'wb'))