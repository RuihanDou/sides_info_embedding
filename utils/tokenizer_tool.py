from tensorflow import keras

def load_tokenzier(path: str) -> keras.preprocessing.text.Tokenizer:
    with open(path, 'r') as file:
        json_str = file.read();
        tokenizer = keras.preprocessing.text.tokenizer_from_json(json_str)
    return tokenizer