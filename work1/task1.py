SYMBOL_TO_MORZE_CODE = {
    'a': '.-', 'b': '-…', 'c': '-.-.', 'd': '-..',
    'e': '.', 'f': '..-.', 'g': '--.', 'h': '….',
    'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..',
    'm': '--', 'n': '-.', 'o': '---', 'p': '.--.',
    'q': '--.-', 'r': '.-.', 's': '…', 't': '-',
    'u': '..-', 'v': '…-', 'w': '.--', 'x': '-..-',
    'y': '-.--', 'z': '--..',
} 


def encode_symbol_in_morze_code(symbol: str) -> str:
    lowercase_symbol = symbol.lower()
    return SYMBOL_TO_MORZE_CODE[lowercase_symbol]


def encode_word_in_morze_code(word: str) -> str:
    return " ".join(map(encode_symbol_in_morze_code, word))


def encode_text_in_morze_code(text: str) -> str:
    words = text.split()
    return "\n".join(map(encode_word_in_morze_code, words))


def main():
    text = input()

    encoded_text = encode_text_in_morze_code(text=text)

    print(encoded_text)


if __name__ == "__main__":
    main()
