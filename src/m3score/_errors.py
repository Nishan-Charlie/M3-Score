"""Exception and warning types raised by m3score."""


class M3ScoreError(RuntimeError):
    """Scoring could not proceed for a reason outside the caller's arguments.

    Examples: the backbone weights cannot be loaded, or pixel data cannot be decoded.
    """


class M3ScoreWarning(UserWarning):
    """Something about the inputs weakens the result without invalidating it."""
