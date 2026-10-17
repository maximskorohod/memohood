"""Profile-aware credential lookup.

hermes >= 0.22 serves every profile from ONE gateway process, so a secondary
profile's ``.env`` never lands in ``os.environ``. ``agent.secret_scope.get_secret``
resolves a key against the active profile's scope; reading ``os.environ``
directly would silently use the default profile's key (or none at all).

Older hermes (no ``agent.secret_scope``) falls back to ``os.environ``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("memohood.secrets")

try:
    from agent.secret_scope import get_secret as _get_secret
except ImportError:  # hermes < 0.22
    _get_secret = None


def get_env(name: str) -> Optional[str]:
    """Credential *name* for the active profile, or ``None`` when unset.

    Fails closed: if hermes refuses an unscoped read (multiplexing on, no
    profile scope bound to this thread) the key is treated as missing, so the
    caller degrades exactly as it does without credentials -- never reading
    another profile's value from ``os.environ``.
    """
    if _get_secret is None:
        return os.environ.get(name) or None
    try:
        return _get_secret(name) or None
    except Exception:  # noqa: BLE001 - UnscopedSecretError: no profile scope on this thread
        logger.debug("memohood: secret %s read outside a profile scope; treating as unset", name, exc_info=True)
        return None
