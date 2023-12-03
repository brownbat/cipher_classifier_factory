# tests/test_ciphers.py

import sys
import unittest
import string
import random
import os

sys.path.append('../')

from ciphers import (
    caesar, vigenere, columnar_transposition, random_noise,
    playfair, beaufort, autokey, _normalize_text, english, _random_keyword)

os.chdir('../')


class TestCiphers(unittest.TestCase):

    def test_caesar(self):
        print("\nTesting caesar ", end='')
        original_text = "Attack at dawn!"
        shift = 3
        # Testing encryption
        encrypted_text = caesar(text=original_text, shift=shift)
        expected_encryption = "Dwwdfn dw gdzq!"
        self.assertEqual(
            encrypted_text,
            expected_encryption,
            "Caesar Cipher encryption failed.")

        # Testing decryption
        decrypted_text = caesar(text=encrypted_text, shift=shift, encode=False)
        self.assertEqual(
            decrypted_text,
            original_text,
            "Caesar Cipher decryption failed.")

    def test_vigenere(self):
        print("\nTesting vigenere ", end='')
        original_text = "attackatdawn"
        keyword = "key"
        encrypted_text = vigenere(text=original_text, key=keyword)
        decrypted_text = vigenere(
            text=encrypted_text, key=keyword, encode=False)
        self.assertEqual(
            decrypted_text,
            original_text,
            "Vigenère Cipher decryption failed.")

    def test_columnar_transposition_encoding(self):
        print("\nTesting columnar known encode ", end='')

        text = "abcdef"
        key = "312"
        expected_encrypted = "becfad"
        encrypted_text = columnar_transposition(text=text, key=key)
        self.assertEqual(
            encrypted_text,
            expected_encrypted,
            "Columnar Transposition encoding failed.")

    def test_columnar_transposition_decoding(self):
        print("\nTesting columnar known decode ", end='')
        encrypted_text = "becfad"
        key = "312"
        expected_decrypted = "abcdef"
        decrypted_text = columnar_transposition(
            text=encrypted_text,
            key=key,
            encode=False)
        self.assertEqual(
            decrypted_text,
            expected_decrypted,
            "Columnar Transposition decoding failed.")

    def test_columnar_transposition_round_trip(self):
        print("\nTesting columnar round trip ", end='')
        original_text = "attackatdawn"
        key = "3214"
        encrypted_text = columnar_transposition(text=original_text, key=key)
        decrypted_text = columnar_transposition(
            text=encrypted_text, key=key, encode=False)
        self.assertEqual(
            decrypted_text,
            original_text,
            "Columnar Transposition round-trip failed.")

    def test_random_noise(self):
        print("\nTesting random noise ", end='')
        length = 10
        noise = random_noise(length)
        self.assertEqual(len(noise), length, "Random Noise incorrect length.")
        for char in noise:
            self.assertIn(
                char, string.ascii_lowercase, "Character not in expected set.")
        length = 100
        self.assertNotEqual(
            random_noise(length), random_noise(length), "Noise not random.")

    def test_playfair_cipher(self):
        print("\nTesting playfair ", end='')
        original_text = "hidethegoldinthegardenx"
        keyword = "keyword"
        encrypted_text = playfair(text=original_text, keyword=keyword)
        decrypted_text = playfair(
            text=encrypted_text, keyword=keyword, encode=False)
        self.assertEqual(
            decrypted_text,
            original_text.replace('j', 'i'),
            "Playfair Cipher decryption failed.")

    def test_playfair_cipher_odd_length(self):
        print("\nTesting playfair with odd length ", end='')
        original_text = "oddlengthtext"
        keyword = "examplekeyword"
        encrypted_text = playfair(text=original_text, keyword=keyword)
        decrypted_text = playfair(
            text=encrypted_text, keyword=keyword, encode=False)
        self.assertEqual(
            decrypted_text,
            original_text.replace('j', 'i'),
            "Playfair Cipher decryption failed for odd length text.")

    def test_playfair_cipher_repeats(self):
        print("\nTesting playfair with repetition ", end='')
        original_text = "aabccdee"
        keyword = "money"
        encrypted_text = playfair(text=original_text, keyword=keyword)
        decrypted_text = playfair(
            text=encrypted_text, keyword=keyword, encode=False)
        self.assertEqual(
            decrypted_text,
            original_text.replace('j', 'i'),
            "Playfair Cipher decryption failed for repeating characters.")

    def test_playfair_cipher_ij(self):
        print("\nTesting playfair with i/j ", end='')
        original_text = "injailjimjimmiesandjigglesjammedjarsofpickles"
        keyword = "jail"
        encrypted_text = playfair(text=original_text, keyword=keyword)
        decrypted_text = playfair(
            text=encrypted_text, keyword=keyword, encode=False)
        self.assertEqual(
            decrypted_text,
            original_text.replace('j', 'i'),
            "Playfair Cipher decryption failed for repeating characters.")

    def test_playfair_cipher_alphabet(self):
        print("\nTesting playfair with entire alphabet ", end='')
        original_text = "thequickbrownfoxjumpedoverthelazydog"
        keyword = "abcdefg"
        encrypted_text = playfair(text=original_text, keyword=keyword)
        decrypted_text = playfair(
            text=encrypted_text, keyword=keyword, encode=False)
        self.assertEqual(
            decrypted_text,
            original_text.replace('j', 'i'),
            "Playfair Cipher decryption failed for repeating characters.")

    def test_beaufort_encryption_decryption(self):
        text = "HelloWorld"
        key = "Key"
        encrypted = beaufort(text=text, key=key)
        decrypted = beaufort(text=encrypted, key=key)
        self.assertEqual(decrypted.lower(), text.lower())

    def test_beaufort_key_repeats(self):
        text = "LongText" * 10  # longer than typical key
        key = "Short"
        encrypted = beaufort(text=text, key=key)
        decrypted = beaufort(text=encrypted, key=key)
        self.assertEqual(decrypted.lower(), text.lower())

    def test_beaufort_non_alphabetic_characters(self):
        text = "Hello, World! 123"
        text = _normalize_text(text)
        key = "Key"
        encrypted = beaufort(text=text, key=key)
        decrypted = beaufort(text=encrypted, key=key)
        self.assertEqual(decrypted.lower(), "HelloWorld".lower())

    def test_beaufort_randomized_text_and_key(self):
        text = ''.join(random.choices(string.ascii_letters, k=50))
        key = ''.join(random.choices(string.ascii_letters, k=10))
        encrypted = beaufort(text=text, key=key)
        decrypted = beaufort(text=encrypted, key=key)
        self.assertEqual(decrypted.lower(), text.lower())

    def test_autokey_basic_encryption_decryption(self):
        text = "HelloWorld"
        key = "Key"
        encrypted = autokey(text=text, key=key, encode=True)
        decrypted = autokey(text=encrypted, key=key, encode=False)
        self.assertEqual(decrypted.lower(), text.lower())

    def test_autokey_key_extension(self):
        text = "LongText" * 5
        key = "Short"
        encrypted = autokey(text=text, key=key, encode=True)
        decrypted = autokey(text=encrypted, key=key, encode=False)
        self.assertEqual(decrypted.lower(), text.lower())

    def test_autokey_non_alphabetic_characters(self):
        text = "Hello, World! 123"
        key = "Key"
        encrypted = autokey(text=text, key=key, encode=True)
        decrypted = autokey(text=encrypted, key=key, encode=False)
        self.assertEqual(decrypted.lower(), "helloworld".lower())

    def test_autokey_case_sensitivity(self):
        text = "HelloWorld"
        key = "Key"
        encrypted = autokey(text=text, key=key, encode=True)
        decrypted = autokey(text=encrypted, key=key.lower(), encode=False)
        self.assertEqual(decrypted.lower(), text.lower())

    def test_autokey_random_key_and_text(self):
        text = english(50)
        key = _random_keyword(10)
        encrypted = autokey(text=text, key=key, encode=True)
        decrypted = autokey(text=encrypted, key=key, encode=False)
        self.assertEqual(decrypted.lower(), text.lower())


if __name__ == '__main__':
    unittest.main()
