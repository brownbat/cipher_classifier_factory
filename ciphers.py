import string
import random
import book_processing
from itertools import cycle
import inspect


def _get_cipher_functions():
    """
    Retrieves cipher functions from this module, excluding helper
    functions, which start with '_'.
    """
    return [func for name, func in globals().items() if inspect.isfunction(func) and not name.startswith('_')]

def _get_cipher_names():
    """
    Retrieves the names of cipher functions from this module, excluding helper
    functions, which start with '_'.
    """
    return [name for name, func in globals().items() if inspect.isfunction(func) and not name.startswith('_')]


def _normalize_text(text):
    """
    Normalize the text by converting to lowercase and keeping only ASCII
    letters.
    """
    return ''.join(_ for _ in text.lower() if _ in string.ascii_lowercase)


def _random_keyword(length=None):
    """
    Generates a random keyword of specified length using English text.
    If length is None, picks a random length between 5 and 20.
    """
    if length is None:
        length = random.randint(5, 20)

    random_keyword = book_processing.get_random_text_passage(length)
    return random_keyword


# TODO deal with the fact these signatures are not uniform
def english(length=None):
    '''
    Returns random strings of english text from leading project gutenberg texts
    '''
    if length is None:
        length = random.randint(200,700)
    return book_processing.get_random_text_passage(length)


def caesar(length=None, text=None, shift=None, encode=True):
    """
    Encrypts or decrypts the text using a Caesar cipher with the specified shift
    If shift is None it is chosen randomly
    If encode is True, the text is encrypted, otherwise it is decrypted.
    """
    if text is None:
        if length is None:
            length = random.randint(200,700)
        text = english(length)
    if shift is None:
        shift = random.randint(1,25)
    def shift_char(c):
        if c.isalpha():
            start = 'A' if c.isupper() else 'a'
            effective_shift = shift if encode else -shift
            return chr(
                (ord(c) - ord(start) + effective_shift) % 26 + ord(start))
        else:
            return c

    return ''.join(shift_char(c) for c in text)


def vigenere(length=None, text=None, key=None, encode=True):
    """
    Encrypts or decrypts the text using a Vigenère cipher with the
    specified key.
    If key is None some random English text is chosen
    If encode is True, the text is encrypted, otherwise it is decrypted.
    """
    if text is None:
        if length is None:
            length = random.randint(200,700)
        text = english(length)
    alphabet = string.ascii_lowercase
    result = []
    if key is None:
        key = _random_keyword()
    key_repeated = ''.join(key for _ in range(len(text) // len(key) + 1))

    for t, k in zip(text, key_repeated):
        if t in alphabet:
            if encode:
                shifted_index = (
                    alphabet.index(t)
                    + alphabet.index(k.lower())) % len(alphabet)
            else:
                shifted_index = (
                    alphabet.index(t)
                    - alphabet.index(k.lower())) % len(alphabet)
            result.append(alphabet[shifted_index])
        else:
            result.append(t)

    return ''.join(result)


def beaufort(length=None, text=None, key=None):
    if text is None:
        if length is None:
            length = random.randint(200,700)
        text = english(length)
    if key is None or key == '':
        key = _random_keyword()

    def shift_char(c, k):
        # Assuming c and k are lowercase letters
        return chr(((ord(k) - ord(c)) % 26) + ord('a'))

    key = key.lower()
    text = text.lower()

    encrypted_text = ''
    for char, key_char in zip(text, cycle(key)):
        encrypted_text += shift_char(char, key_char)

    return encrypted_text


def autokey(length=None, text=None, key=None, encode=True):
    if text is None:
        if length is None:
            length = random.randint(200, 700)
        text = english(length)
    if key is None or key == '':
        key = _random_keyword()
    text = _normalize_text(text)

    def shift_char(c, k):
        if encode:
            return chr(((ord(c) - ord('a') + ord(k) - ord('a')) % 26) + ord('a'))
        else:
            return chr(((ord(c) - ord(k) + 26) % 26) + ord('a'))

    key = _normalize_text(key)
    if encode:
        key = (key + text)[:len(text)]  # Extend or truncate key for encoding

    result = ''
    key_index = 0

    for char in text:
        key_char = key[key_index % len(key)]
        result_char = shift_char(char, key_char)
        result += result_char
        key_index += 1
        if not encode:
            key += result_char  # Append decrypted character to key

    return result


def random_noise(length=None, characters=string.ascii_lowercase):
    """
    Generates a string of random characters of the specified length drawning
    from `characters`.
    """
    if length is None:
        length = random.randint(200,700)
    return ''.join(random.choice(characters) for _ in range(length))


def _playfair_build_matrix(keyword=None):
    if keyword is None:
        keyword = _random_keyword()
    matrix = ['' for _ in range(5)]
    seen = set()
    i, j = 0, 0
    for char in keyword + string.ascii_lowercase:
        if char not in seen and char != 'j':
            matrix[i] += char
            seen.add(char)
            j += 1
            if j == 5:
                i += 1
                j = 0
    return matrix


def _playfair_digraph_prep(text, encode=True):
    result = []
    i = 0
    while i < len(text):
        char1 = text[i]
        char2 = 'x' if i + 1 == len(text) else text[i + 1]

        if encode and char1 == char2:
            result.append(char1)
            result.append('x')
            i += 1  # Increment i by 1 to process the next character in the next iteration
        else:
            result.append(char1)
            result.append(char2)
            i += 2  # Increment i by 2 as we've processed two characters

    return ''.join(result)


def _playfair_decrypt_postprocess(text):
    result = []
    i = 0
    while i < len(text):
        if i < len(text) - 2 and text[i] == text[i + 2] and text[i + 1] == 'x':
            result.append(text[i])
            i += 2  # Skip over the 'x'
        elif i == len(text) - 1 and text[i] == 'x':
            i += 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def _playfair_encrypt_decrypt(matrix, digraph, encode):
    def find_position(letter):
        if letter == 'j':
            letter = 'i'
        for i, row in enumerate(matrix):
            if letter in row:
                return i, row.index(letter)
        return None, None

    def shift_row(row, col, shift):
        return matrix[row][(col + shift) % 5]

    def shift_column(row, col, shift):
        return matrix[(row + shift) % 5][col]

    encrypted_digraph = ''
    for i in range(0, len(digraph), 2):
        char1, char2 = digraph[i], digraph[i + 1]
        row1, col1 = find_position(char1)
        row2, col2 = find_position(char2)

        # Diagnostic print statement
        if row1 is None or col1 is None or row2 is None or col2 is None:
            print(
                f"Error: One of the positions is None. char1: {char1}, "
                + f"char2: {char2}, row1: {row1}, col1: {col1}, row2: {row2},"
                + f" col2: {col2}")

        shift = 1 if encode else -1
        if row1 == row2:
            encrypted_digraph += shift_row(row1, col1, shift)
            encrypted_digraph += shift_row(row2, col2, shift)
        elif col1 == col2:
            encrypted_digraph += shift_column(row1, col1, shift)
            encrypted_digraph += shift_column(row2, col2, shift)
        else:
            encrypted_digraph += matrix[row1][col2]
            encrypted_digraph += matrix[row2][col1]
    if not encode:
        encrypted_digraph = _playfair_decrypt_postprocess(encrypted_digraph)

    return encrypted_digraph
    

def playfair(length=None, text=None, keyword=None, encode=True):
    if text is None:
        if length is None:
            length = random.randint(200, 700)
        text = english(length)
    text = _playfair_digraph_prep(text, encode)
    text = _normalize_text(text)
    if keyword is None:
        keyword = _random_keyword()
    matrix = _playfair_build_matrix(keyword)

    return _playfair_encrypt_decrypt(matrix, text, encode)


def columnar_transposition(length=None, text=None, key=None, padding_char=None, encode=True):
    """
    Encrypts or decrypts text using columnar transposition using key.

    For encryption (encode=True), normalizes the text and then arranges it into
    a matrix based on the length of the key. If the text does not perfectly
    fit the matrix, it is padded. The padding character can be specified with
    the 'padding_char' argument. If 'padding_char' is None, a random lowercase
    letter is chosen. The function returns the text read column-wise based on
    the order of the key, e.g., key 312 reads the second column first, then the
    third, then the first.

    For decryption (encode=False), the function reconstructs the original
    message from the columns based on the key order. Any extra padding at the
    end of the message is stripped.

    Parameters:
        text (str): The text to be encrypted or decrypted.
        key (str): The key used for the columnar transposition. The length of
            the key determines the number of columns in the matrix.
        encode (bool, optional): If True, the function encrypts the text. If
            False, it decrypts the text. Defaults to True.
        padding_char (str, optional): The character used for padding if the text
            does not perfectly fit into the matrix. If None,
            a random lowercase letter is used. Defaults to None.

    Returns:
        str: The encrypted or decrypted text.
    """
    if text is None:
        if length is None:
            length = random.randint(200,700)
        text = english(length)
    # Normalize the text
    text = _normalize_text(text)
    if key is None:
        key = _random_keyword()

    num_columns = len(key)
    num_rows = len(text) // num_columns + (1 if len(text) % num_columns else 0)

    # Initialize the matrix
    matrix = ['' for _ in range(num_rows)]

    if encode:
        # Fill the matrix row-wise
        for i, char in enumerate(text):
            matrix[i // num_columns] += char

        if padding_char is None:
            padding_char = random.choice(string.ascii_lowercase)
        matrix[-1] += padding_char * (
            num_columns
            - len(matrix[-1]))  # Pad the last row if necessary

        # Read the columns based on the key order
        key_indices = sorted(range(len(key)), key=lambda x: key[x])
        return ''.join(
            matrix[row][index]
            for index in key_indices for row in range(num_rows))
    else:
        # Decoding
        # Calculate column lengths
        col_lengths = [
            len(text)
            // num_columns
            + (1 if i < len(text) % num_columns else 0)
            for i in range(num_columns)]

        # Determine the order to read columns from the encrypted text
        ordered_key = sorted(enumerate(key), key=lambda x: x[1])

        # Create a list to store the start index of each column in the ct
        idx = 0
        col_starts = [0] * num_columns
        for i, _ in ordered_key:
            col_starts[i] = idx
            idx += col_lengths[i]

        # Distribute the text into columns based on the key
        for i, start in enumerate(col_starts):
            column_length = col_lengths[i]
            matrix_column = text[start:start + column_length]
            for j in range(column_length):
                matrix[j] += matrix_column[j] if j < len(matrix_column) else ' '

        # Read the rows to reconstruct the original message
        return ''.join(''.join(row) for row in matrix).rstrip()


# TODO implement fractionated morse, bifid, ADFGVX, trifid, VIC, enigma


if __name__ == "__main__":
    print("Testing random English text:")
    print(english(50))
    
    print("\nTesting Random Noise Generator:")
    print(random_noise(50))

    # Test the Caesar Cipher
    print("\nTesting Caesar Cipher:")
    original_text = "Attack at dawn!"
    shift = 3
    encrypted = caesar(text=original_text, shift=shift)
    decrypted = caesar(text=encrypted, shift=shift, encode=False)
    print(
        f"Original: {original_text},"
        + f" Encrypted: {encrypted}, Decrypted: {decrypted}")

    # Test the Vigenère Cipher
    print("\nTesting Vigenère Cipher:")
    keyword = "KEY"
    encrypted = vigenere(text=original_text, key=keyword)
    decrypted = vigenere(text=encrypted, key=keyword, encode=False)
    print(
        f"Original: {original_text},"
        + f" Encrypted: {encrypted}, Decrypted: {decrypted}")

    # Test the Columnar Transposition Cipher
    print("\nTesting Columnar Transposition Cipher:")
    key = "3214"
    encrypted = columnar_transposition(text=original_text, key=key)
    decrypted = columnar_transposition(text=encrypted, key=key, encode=False)
    print(
        f"Original: {original_text},"
        + f" Encrypted: {encrypted}, Decrypted: {decrypted}")

    original_text = "Playfair is a cipher"
    key = _random_keyword()
    encrypted = playfair(text=original_text, keyword=key)
    decrypted = playfair(text=encrypted, encode=False, keyword=key)
    print("\nTesting Playfair")
    print(
        f"Original: {original_text},"
        + f"Encrypted: {encrypted}, Decrypted: {decrypted}"
        )

