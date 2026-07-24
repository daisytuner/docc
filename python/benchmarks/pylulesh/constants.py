# See LICENSE.md for full license.
"""
Constants used in computation.
"""

# Boundary conditions
# 2 BCs on each of 6 hexahedral faces (12 bits)
XI = {
    "M": {
        "mask": 0x00007,  # Equivalent to XI_M
        "SYMM": 0x00001,  # Equivalent to XI_M_SYMM
        "FREE": 0x00002,  # Equivalent to XI_M_FREE
        "COMM": 0x00004,  # Equivalent to XI_M_COMM
    },
    "P": {
        "mask": 0x00038,
        "SYMM": 0x00008,
        "FREE": 0x00010,
        "COMM": 0x00020,
    },
}

ETA = {
    "M": {
        "mask": 0x001C0,
        "SYMM": 0x00040,
        "FREE": 0x00080,
        "COMM": 0x00100,
    },
    "P": {
        "mask": 0x00E00,
        "SYMM": 0x00200,
        "FREE": 0x00400,
        "COMM": 0x00800,
    },
}

ZETA = {
    "M": {
        "mask": 0x07000,
        "SYMM": 0x01000,
        "FREE": 0x02000,
        "COMM": 0x04000,
    },
    "P": {
        "mask": 0x38000,
        "SYMM": 0x08000,
        "FREE": 0x10000,
        "COMM": 0x20000,
    },
}
