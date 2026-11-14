import json
import math
import os
import struct
import tempfile
import unittest

from binjson.codec import (
    ARRAY, BIGINT, DecodeError, FLOAT64, INT16, INT32, INT64, INT8, NULL, OBJECT,
    STRING, TRUE, decode, decode_file, encode, encode_file, read_varint,
    size_report, write_varint,
)


class VarintTest(unittest.TestCase):
    def test_round_trip(self):
        for value in (0, 1, 127, 128, 300, 2 ** 20, 2 ** 63):
            out = bytearray()
            write_varint(out, value)
            self.assertEqual(read_varint(bytes(out), 0), (value, len(out)))

    def test_boundary_is_two_bytes(self):
        out = bytearray()
        write_varint(out, 128)
        self.assertEqual(len(out), 2)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            write_varint(bytearray(), -1)

    def test_truncated_fails(self):
        with self.assertRaises(DecodeError):
            read_varint(b"\x80", 0)


class ScalarTest(unittest.TestCase):
    def test_nulls_and_booleans_are_one_byte(self):
        self.assertEqual(encode(None), bytes([NULL]))
        self.assertEqual(encode(True), bytes([TRUE]))
        self.assertEqual(len(encode(False)), 1)

    def test_integer_widths(self):
        self.assertEqual(encode(1)[0], INT8)
        self.assertEqual(encode(300)[0], INT16)
        self.assertEqual(encode(70000)[0], INT32)
        self.assertEqual(encode(2 ** 40)[0], INT64)

    def test_big_integers_survive(self):
        value = 10 ** 40
        self.assertEqual(encode(value)[0], BIGINT)
        self.assertEqual(decode(encode(value)), value)

    def test_floats(self):
        for value in (0.0, -0.0, 1.5, 3.4e300, float("inf")):
            blob = encode(value)
            self.assertEqual(blob[0], FLOAT64)
            back = decode(blob)
            self.assertEqual(math.copysign(1, back), math.copysign(1, value))

    def test_nan_round_trip(self):
        self.assertTrue(math.isnan(decode(encode(float("nan")))))

    def test_unicode_string(self):
        value = "salam \u062f\u0646\u06cc\u0627 \U0001F600"
        self.assertEqual(decode(encode(value)), value)
        self.assertEqual(encode(value)[0], STRING)


class ContainerTest(unittest.TestCase):
    def test_array_and_object_tags(self):
        self.assertEqual(encode([])[0], ARRAY)
        self.assertEqual(encode({})[0], OBJECT)

    def test_nested_round_trip(self):
        value = {"a": [1, 2, {"b": None}], "c": {"d": [True, False]}, "e": "x"}
        self.assertEqual(decode(encode(value)), value)

    def test_key_order_is_preserved(self):
        value = {"z": 1, "a": 2, "m": 3}
        self.assertEqual(list(decode(encode(value)).keys()), ["z", "a", "m"])

    def test_non_string_keys_rejected(self):
        with self.assertRaises(TypeError):
            encode({1: "x"})

    def test_unsupported_type_rejected(self):
        with self.assertRaises(TypeError):
            encode({1, 2, 3})

    def test_empty_containers_are_two_bytes(self):
        self.assertEqual(len(encode([])), 2)
        self.assertEqual(len(encode({})), 2)


class ErrorTest(unittest.TestCase):
    def test_trailing_bytes_rejected(self):
        with self.assertRaises(DecodeError):
            decode(encode(1) + b"\x00")

    def test_unknown_tag_rejected(self):
        with self.assertRaises(DecodeError):
            decode(b"\x7f")

    def test_truncated_string_rejected(self):
        with self.assertRaises(DecodeError):
            decode(bytes([STRING, 5]) + b"ab")

    def test_truncated_integer_rejected(self):
        with self.assertRaises(DecodeError):
            decode(bytes([INT32, 0, 0]))

    def test_empty_input_rejected(self):
        with self.assertRaises(DecodeError):
            decode(b"")


class SizeTest(unittest.TestCase):
    def test_smaller_than_json_for_numbers(self):
        value = {"ports": list(range(100)), "name": "web"}
        text_bytes, binary_bytes = size_report(value)
        self.assertLess(binary_bytes, text_bytes)

    def test_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "payload.binjson")
            encode_file({"x": [1, 2, 3]}, path)
            self.assertEqual(decode_file(path), {"x": [1, 2, 3]})
            self.assertEqual(os.path.getsize(path), len(encode({"x": [1, 2, 3]})))

    def test_matches_json_shape(self):
        value = {"a": 1, "b": [None, True, "s"], "c": {"d": 2.5}}
        self.assertEqual(json.loads(json.dumps(value)), decode(encode(value)))


if __name__ == "__main__":
    unittest.main()
