"""Math functionality."""

__all__ = ['round_up']

def round_up(value):
    """ Rounds value to the next integer.

    Args:
        value (number): number to round
    """
    return int(value + 0.5)
