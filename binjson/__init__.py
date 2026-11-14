"""binjson - a compact binary encoding for JSON-shaped data."""

from .codec import DecodeError, decode, decode_file, encode

__all__ = ["DecodeError", "decode", "decode_file", "encode"]
