"""Variant-free binary encoding for the JSON data model.

Design constraints, in priority order:

1. Every value carries its type in one tag byte, so a decoder needs no schema.
2. Small integers, short strings and short arrays cost less than their JSON text.
3. No length field is bigger than it has to be: lengths use a varint.
4. Round-trip is exact for anything json.dumps can emit, including -0.0 and big ints.

Layout:

    0x00 null   0x01 false   0x02 true
    0x03 int8   0x04 int16   0x05 int32   0x06 int64   0x07 bigint (varint length + digits)
    0x08 float64 (8 bytes, big-endian)
    0x09 string (varint length + utf-8)
    0x0A array  (varint count + values)
    0x0B object (varint count + (key, value) pairs, keys are strings)

A varint is little-endian 7 bits per byte with the high bit as continuation.
Integers are stored two's complement at the smallest width that fits.
"""
import json
import math
import struct

NULL, FALSE, TRUE = 0x00, 0x01, 0x02
INT8, INT16, INT32, INT64, BIGINT = 0x03, 0x04, 0x05, 0x06, 0x07
FLOAT64 = 0x08
STRING, ARRAY, OBJECT = 0x09, 0x0A, 0x0B

INT_RANGES = [
    (INT8, -(2 ** 7), 2 ** 7 - 1, "!b"),
    (INT16, -(2 ** 15), 2 ** 15 - 1, "!h"),
    (INT32, -(2 ** 31), 2 ** 31 - 1, "!i"),
    (INT64, -(2 ** 63), 2 ** 63 - 1, "!q"),
]


class DecodeError(Exception):
    pass


def write_varint(out, value):
    if value < 0:
        raise ValueError("varint cannot be negative")
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def read_varint(buf, offset):
    result = 0
    shift = 0
    while True:
        if offset >= len(buf):
            raise DecodeError("varint runs past the end of the buffer")
        byte = buf[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
        shift += 7
        if shift > 70:
            raise DecodeError("varint is longer than 10 bytes")


def _encode_scalar(value, out):
    if value is None:
        out.append(NULL)
        return
    if value is True:
        out.append(TRUE)
        return
    if value is False:
        out.append(FALSE)
        return
    if isinstance(value, int):
        for tag, low, high, fmt in INT_RANGES:
            if low <= value <= high:
                out.append(tag)
                out.extend(struct.pack(fmt, value))
                return
        digits = str(value).encode("ascii")
        out.append(BIGINT)
        write_varint(out, len(digits))
        out.extend(digits)
        return
    if isinstance(value, float):
        out.append(FLOAT64)
        out.extend(struct.pack("!d", value))
        return
    if isinstance(value, str):
        raw = value.encode("utf-8")
        out.append(STRING)
        write_varint(out, len(raw))
        out.extend(raw)
        return
    raise TypeError("cannot encode %r" % type(value).__name__)


def encode(value):
    out = bytearray()
    _encode_value(value, out)
    return bytes(out)


def _encode_value(value, out):
    if isinstance(value, (list, tuple)):
        out.append(ARRAY)
        write_varint(out, len(value))
        for item in value:
            _encode_value(item, out)
        return
    if isinstance(value, dict):
        out.append(OBJECT)
        write_varint(out, len(value))
        for key in value:
            if not isinstance(key, str):
                raise TypeError("object keys must be strings, got %r" % type(key).__name__)
            raw = key.encode("utf-8")
            write_varint(out, len(raw))
            out.extend(raw)
            _encode_value(value[key], out)
        return
    _encode_scalar(value, out)


def decode(data):
    value, offset = _decode_value(memoryview(data), 0)
    if offset != len(data):
        raise DecodeError("%d trailing bytes after the root value" % (len(data) - offset))
    return value


def _decode_value(buf, offset):
    if offset >= len(buf):
        raise DecodeError("unexpected end of data")
    tag = buf[offset]
    offset += 1
    if tag == NULL:
        return None, offset
    if tag == TRUE:
        return True, offset
    if tag == FALSE:
        return False, offset
    for kind, _low, _high, fmt in INT_RANGES:
        if tag == kind:
            size = struct.calcsize(fmt)
            if offset + size > len(buf):
                raise DecodeError("truncated integer")
            return struct.unpack(fmt, buf[offset:offset + size])[0], offset + size
    if tag == BIGINT:
        length, offset = read_varint(buf, offset)
        if offset + length > len(buf):
            raise DecodeError("truncated bigint")
        return int(bytes(buf[offset:offset + length]).decode("ascii")), offset + length
    if tag == FLOAT64:
        if offset + 8 > len(buf):
            raise DecodeError("truncated float")
        return struct.unpack("!d", buf[offset:offset + 8])[0], offset + 8
    if tag == STRING:
        length, offset = read_varint(buf, offset)
        if offset + length > len(buf):
            raise DecodeError("truncated string")
        return bytes(buf[offset:offset + length]).decode("utf-8"), offset + length
    if tag == ARRAY:
        count, offset = read_varint(buf, offset)
        items = []
        for _ in range(count):
            item, offset = _decode_value(buf, offset)
            items.append(item)
        return items, offset
    if tag == OBJECT:
        count, offset = read_varint(buf, offset)
        result = {}
        for _ in range(count):
            length, offset = read_varint(buf, offset)
            if offset + length > len(buf):
                raise DecodeError("truncated object key")
            key = bytes(buf[offset:offset + length]).decode("utf-8")
            offset += length
            result[key], offset = _decode_value(buf, offset)
        return result, offset
    raise DecodeError("unknown tag 0x%02x at offset %d" % (tag, offset - 1))


def encode_file(value, path):
    with open(path, "wb") as fh:
        fh.write(encode(value))


def decode_file(path):
    with open(path, "rb") as fh:
        return decode(fh.read())


def size_report(value):
    """Return (json_bytes, binjson_bytes) so callers can see what they saved."""
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False,
                      default=_reject_float_noise).encode("utf-8")
    return len(text), len(encode(value))


def _reject_float_noise(value):
    raise TypeError("json.dumps cannot represent %r" % (value,))


def is_finite(value):
    return not isinstance(value, float) or math.isfinite(value)
