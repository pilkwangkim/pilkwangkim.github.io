#!/usr/bin/env python3
"""Small, exact binary codes used by the error-correcting codes article.

No packages are required. Coordinates use column vectors: c = G m and H c = 0,
with every addition and multiplication evaluated modulo 2. The seven positions
are [p1, p2, d1, p4, d2, d3, d4], counted from 1 at the left.

An ordinary Hamming decoder assumes at most one flipped bit. It cannot verify
that assumption: two errors lead to a different valid codeword. The extended
decoder's SECDED classification is guaranteed only when at most two bits flip.

Reference: MIT 6.02, chapter 6, "Coping with Bit Errors Using Error Correction
Codes": https://web.mit.edu/6.02/www/f2011/handouts/6.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, isfinite
from typing import Iterable, Sequence, Union

BitInput = Union[str, Iterable[int]]
Word = tuple[int, ...]

H = (
    (1, 0, 1, 0, 1, 0, 1),
    (0, 1, 1, 0, 0, 1, 1),
    (0, 0, 0, 1, 1, 1, 1),
)

# Each column is the codeword produced by one unit message vector.
# Data coordinates 3, 5, 6, 7 reproduce m, so encoding is injective.
G = (
    (1, 1, 0, 1),
    (1, 0, 1, 1),
    (1, 0, 0, 0),
    (0, 1, 1, 1),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
)

DATA_INDICES = (2, 4, 5, 6)  # Zero-based Python indices, not position labels.


def bits(value: BitInput, length: int | None = None) -> Word:
    """Validate a 0/1 string or an iterable of integer bits; return a tuple.

    Whitespace, floats, and Boolean values are rejected rather than silently
    converted. Inputs are copied, so decoders never mutate the caller's word.
    """
    if isinstance(value, str):
        if any(char not in "01" for char in value):
            raise ValueError("A bit string may contain only '0' and '1'.")
        result = tuple(int(char) for char in value)
    else:
        try:
            result = tuple(value)
        except TypeError as error:
            raise ValueError("Expected a bit string or an iterable of bits.") from error
        if any(type(bit) is not int or bit not in (0, 1) for bit in result):
            raise ValueError("Each bit must be the integer 0 or 1.")
    if length is not None and len(result) != length:
        raise ValueError(f"Expected {length} bits, received {len(result)}.")
    return result


def bitstring(value: BitInput) -> str:
    """Print positions in their left-to-right order without tuple punctuation."""
    return "".join(str(bit) for bit in bits(value))


def xor_vectors(left: BitInput, right: BitInput) -> Word:
    """Add equal-length binary vectors in F2 (componentwise XOR)."""
    a = bits(left)
    b = bits(right, len(a))
    return tuple(x ^ y for x, y in zip(a, b))


def matvec(matrix: Sequence[BitInput], vector: BitInput) -> Word:
    """Multiply a binary matrix by a column vector, reducing each sum mod 2."""
    v = bits(vector)
    rows = tuple(bits(row, len(v)) for row in matrix)
    return tuple(sum(a * b for a, b in zip(row, v)) % 2 for row in rows)


def flip(word: BitInput, *positions: int) -> Word:
    """Toggle the selected one-based positions; repeated toggles cancel."""
    result = list(bits(word))
    for position in positions:
        if type(position) is not int or not 1 <= position <= len(result):
            raise ValueError("Each position must be an integer from 1 to n.")
        result[position - 1] ^= 1
    return tuple(result)


def hamming_distance(left: BitInput, right: BitInput) -> int:
    """Count unequal coordinates; no geometric embedding is needed."""
    return sum(xor_vectors(left, right))


def encode_hamming(message: BitInput) -> Word:
    """Encode four data bits as the systematic Hamming (7,4) word G m."""
    return matvec(G, bits(message, 4))


def syndrome(received: BitInput) -> Word:
    """Return H r in the row order (s1, s2, s4), NOT most-significant first."""
    return matvec(H, bits(received, 7))


def syndrome_position(received: BitInput) -> int:
    """Interpret (s1,s2,s4) as s1 + 2*s2 + 4*s4; zero means H r = 0."""
    s1, s2, s4 = syndrome(received)
    return s1 + 2 * s2 + 4 * s4


@dataclass(frozen=True)
class DecodeResult:
    """A decoder decision, not independent knowledge of the transmitted word.

    `word` and `message` are the decoder's candidates; both are None when the
    extended decoder refuses to correct a detected double error. A zero
    syndrome says only that the received word satisfies the parity equations.
    """

    word: Word | None
    message: Word | None
    syndrome: Word
    status: str
    flipped_position: int | None = None
    overall_parity: int | None = None


def decode_hamming(received: BitInput) -> DecodeResult:
    """Decode under the assumption of zero or one bit error.

    For more errors the candidate may be wrong, even though H * candidate = 0.
    In particular, this decoder does NOT detect-and-refuse double errors.
    """
    r = bits(received, 7)
    s = syndrome(r)
    position = s[0] + 2 * s[1] + 4 * s[2]
    candidate = flip(r, position) if position else r
    return DecodeResult(
        word=candidate,
        message=tuple(candidate[index] for index in DATA_INDICES),
        syndrome=s,
        status="corrected_assuming_one_error" if position else "no_error_detected",
        flipped_position=position or None,
    )


def encode_secded(message: BitInput) -> Word:
    """Append an overall even-parity bit: an extended Hamming (8,4) word."""
    word = encode_hamming(message)
    return word + (sum(word) % 2,)


def decode_secded(received: BitInput) -> DecodeResult:
    """Single-error correction / double-error detection for at most two flips.

    The last bit enforces even parity across all eight positions. Under the
    stated bound, a nonzero Hamming syndrome with even total parity identifies
    a double error, which must be reported instead of 'corrected'. Three or
    more flips can invalidate the classification and may be miscorrected.
    """
    r = bits(received, 8)
    s = syndrome(r[:7])
    position = s[0] + 2 * s[1] + 4 * s[2]
    overall = sum(r) % 2

    if position and not overall:
        return DecodeResult(None, None, s, "detected_double_error", None, overall)
    if position:
        candidate = flip(r, position)
        status = "corrected_code_bit"
    elif overall:
        position = 8
        candidate = flip(r, 8)
        status = "corrected_overall_parity"
    else:
        candidate = r
        status = "no_error_detected"
    return DecodeResult(
        candidate,
        tuple(candidate[index] for index in DATA_INDICES),
        s,
        status,
        position or None,
        overall,
    )


def decode_single(received: BitInput) -> Word:
    """Reference solution for the coding exercise: return four decoded bits.

    Input promise: seven bits from this Hamming encoder and at most one flip.
    XORing the one-based indices whose bits are 1 computes the syndrome index
    because those indices are precisely the binary columns of H.
    """
    candidate = list(bits(received, 7))
    error_position = 0
    for position, bit in enumerate(candidate, start=1):
        if bit:
            error_position ^= position
    if error_position:
        candidate[error_position - 1] ^= 1
    return tuple(candidate[index] for index in DATA_INDICES)


def repetition_failure_probability(p: float, repeats: int = 3) -> float:
    """Majority-decoding failure on independent flips of probability p.

    Repeats must be odd. This compares one information bit's error probability,
    not equal-rate or equal-energy communication schemes. Correlated flips do
    not obey this binomial calculation.
    """
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not isfinite(p):
        raise ValueError("p must be a finite number between 0 and 1.")
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1.")
    if type(repeats) is not int or repeats < 1 or repeats % 2 == 0:
        raise ValueError("repeats must be a positive odd integer.")
    return sum(
        comb(repeats, errors) * p**errors * (1 - p) ** (repeats - errors)
        for errors in range(repeats // 2 + 1, repeats + 1)
    )


def main() -> None:
    message = bits("1011")
    sent = encode_hamming(message)
    received = flip(sent, 5)
    decoded = decode_hamming(received)
    print("message:", bitstring(message))
    print("sent:   ", bitstring(sent))
    print("received:", bitstring(received))
    print("syndrome (s1,s2,s4):", decoded.syndrome)
    print("position:", decoded.flipped_position)
    print("decoded:", bitstring(decoded.message))

    # These two errors fool an ordinary Hamming decoder.
    double_received = flip(sent, 3, 5)
    wrong = decode_hamming(double_received)
    print("double-error candidate:", bitstring(wrong.message))
    extended = decode_secded(flip(encode_secded(message), 3, 5))
    print("extended decoder:", extended.status)


if __name__ == "__main__":
    main()
