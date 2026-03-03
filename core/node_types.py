from enum import IntEnum


class Nodetype(IntEnum):
    """Shared Nodetype enum to avoid circular imports."""

    Video = 0
    Security = 1
    IoT = 2
