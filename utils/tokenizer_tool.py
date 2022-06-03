from tensorflow import keras

def load_tokenizer(path: str) -> keras.preprocessing.text.Tokenizer:
    """
    从 txt 文件存储的 json 字符串中解析 tokenizer
    本项目 tokenizer 分的词是 side information 的各种tag
    :param path:
    :return:
    """
    with open(path, 'r') as file:
        json_str = file.read();
        tokenizer = keras.preprocessing.text.tokenizer_from_json(json_str)
    return tokenizer