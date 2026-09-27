"""Locations used by the archived confirmation tools.

The public result verifier does not require the private acquisition workspace.
"""
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def private_workspace():
    """Return the explicitly configured acquisition and grading workspace."""
    value = os.environ.get('WCL_PRIVATE_WORKSPACE')
    if not value:
        raise RuntimeError(
            'Set WCL_PRIVATE_WORKSPACE to the private acquisition and grading workspace. '
            'Published-result verification does not need this workspace.')
    return Path(value).expanduser().resolve()
