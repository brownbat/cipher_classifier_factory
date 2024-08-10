# TODO - continue testing and refining

def generate_keyed_alphabet(key):
    standard_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    key = ''.join(sorted(set(key), key=key.index))  # Remove duplicates while preserving order
    keyed_alphabet = key.upper() + ''.join([char for char in standard_alphabet if char not in key])
    return keyed_alphabet

def vigenere_cipher(text, key, custom_alphabet=None):
    if custom_alphabet:
        alphabet = custom_alphabet
    else:
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    n = len(alphabet)
    
    # Ensure both text and key are in uppercase
    text = text.upper()
    key = key.upper()
    
    # Result variable
    result = ''
    
    # Perform the cipher operation
    key_index = 0  # Initial key index
    for char in text:
        if char in alphabet:
            text_index = alphabet.index(char)
            key_char = key[key_index % len(key)]
            key_index += 1
            key_index = alphabet.index(key_char)
            
            # Vigenère cipher: result_char = (text_char + key_char) mod 26
            result_index = (text_index + key_index) % n
            result_char = alphabet[result_index]
            result += result_char
        else:
            result += char
            
    return result

# Test cases
tests = [
    {"pt": "ATTACKATDAWN", "key": "SECRET", "abc_key": "KRYPTOS", "ct": "GIGBQTGINBBW"},
    # Add more test cases here
]

# Run the tests
for test in tests:
    # Generate the keyed alphabet for the current test case
    keyed_alphabet = generate_keyed_alphabet(test['abc_key'])
    
    # Encrypt the plaintext using the key and the keyed alphabet
    encrypted_text = vigenere_cipher(test['pt'], test['key'], keyed_alphabet)
    
    # Check if the encrypted text matches the expected ciphertext
    if encrypted_text == test['ct']:
        print(f"Test passed: Plaintext '{test['pt']}' encrypted with key '{test['key']}' and alphabet key '{test['abc_key']}' gives ciphertext '{encrypted_text}' as expected.")
    else:
        print(f"Test failed: Plaintext '{test['pt']}' encrypted with key '{test['key']}' and alphabet key '{test['abc_key']}' gives ciphertext '{encrypted_text}', but expected '{test['ct']}'.")
