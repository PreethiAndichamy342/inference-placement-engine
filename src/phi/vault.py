"""
PHI Vault — encrypted in-memory store for de-identified entity maps.

After a DeIdentifier replaces PHI tokens, the original values must be stored
somewhere safe so they can be re-attached to the inference response.  PHIVault
encrypts the entity_map with AES-256-GCM (via Fernet, which wraps AES-128-CBC
+ HMAC-SHA256 — see note below) and keeps the ciphertext in memory, keyed by
request_id.

Key management
--------------
The vault key is derived from the ``PHI_VAULT_KEY`` environment variable.  If
the variable is absent, a random 32-byte key is generated at startup and a
warning is logged — sufficient for a demo but not suitable for production where
the key must survive process restarts.

Fernet vs AES-256-GCM
---------------------
The Python ``cryptography`` library's :class:`~cryptography.fernet.Fernet`
uses AES-128-CBC + HMAC-SHA256 under the hood, not AES-256-GCM.  For true
AES-256-GCM, this module uses
:class:`~cryptography.hazmat.primitives.ciphers.aead.AESGCM` directly with a
256-bit key.  The Fernet class is kept as a fallback for key generation
utilities only.

Thread safety
-------------
A ``threading.Lock`` serialises all dict mutations so the vault is safe to use
from concurrent FastAPI request handlers.

Usage
-----
    vault = PHIVault()
    vault.store("req-001", {"<PERSON_1>": "Jane Doe", "<SSN_1>": "123-45-6789"})
    entity_map = vault.retrieve("req-001")   # decrypts + removes
    vault.store("req-002", {...})
    vault.delete("req-002")                  # explicit removal without decryption
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import threading
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

_NONCE_BYTES = 12   # 96-bit nonce recommended for AES-GCM
_KEY_BYTES   = 32   # 256-bit key


def _load_or_generate_key() -> bytes:
    """
    Derive a 32-byte AES key.

    Priority:
    1. ``PHI_VAULT_KEY`` env var — expected as a URL-safe base64-encoded 32-byte value.
    2. Generate a fresh random key (suitable for demos / single-process deployments).

    Logs a WARNING when a generated key is used, since it will be lost on restart.
    """
    raw = os.getenv("PHI_VAULT_KEY", "")
    if raw:
        try:
            key = base64.urlsafe_b64decode(raw.encode())
            if len(key) != _KEY_BYTES:
                raise ValueError(f"Expected 32 bytes, got {len(key)}")
            logger.info("PHIVault: loaded key from PHI_VAULT_KEY env var")
            return key
        except Exception as exc:
            logger.error(
                "PHIVault: invalid PHI_VAULT_KEY (%s) — generating ephemeral key", exc
            )

    key = secrets.token_bytes(_KEY_BYTES)
    logger.warning(
        "PHIVault: PHI_VAULT_KEY not set — using ephemeral key. "
        "Entity maps will be unrecoverable after process restart."
    )
    return key


class PHIVault:
    """
    Encrypted in-memory store for PHI entity maps.

    Each entity map is JSON-serialised, then encrypted with AES-256-GCM using
    a fresh random 12-byte nonce per call.  The nonce is prepended to the
    ciphertext before storage so that :meth:`retrieve` can extract it.

    Args:
        key : Optional raw 32-byte key.  When ``None`` (default) the key is
              loaded from ``PHI_VAULT_KEY`` or generated randomly.
        ttl_entries : Maximum number of entries retained in memory (oldest
                      entries are evicted when the limit is exceeded).
                      Default 10 000.
    """

    def __init__(
        self,
        key: Optional[bytes] = None,
        ttl_entries: int = 10_000,
    ) -> None:
        self._key  = key if key is not None else _load_or_generate_key()
        self._aesgcm = AESGCM(self._key)
        self._store: dict[str, bytes] = {}          # request_id → nonce+ciphertext
        self._order: list[str]        = []           # insertion order for eviction
        self._lock  = threading.Lock()
        self._ttl   = ttl_entries

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def store(self, request_id: str, entity_map: dict[str, str]) -> None:
        """
        Encrypt *entity_map* and store it under *request_id*.

        If *entity_map* is empty, nothing is stored (avoids wasting memory on
        no-op de-identification calls).

        Raises
        ------
        RuntimeError
            If encryption fails (e.g., corrupt key).  Callers should treat this
            as a fatal error and reject the request rather than proceeding with
            unencrypted data.
        """
        if not entity_map:
            return

        plaintext = json.dumps(entity_map, ensure_ascii=False).encode()
        nonce     = secrets.token_bytes(_NONCE_BYTES)

        try:
            ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data=None)
        except Exception as exc:
            raise RuntimeError(f"PHIVault: encryption failed for request {request_id}") from exc

        blob = nonce + ciphertext  # 12-byte nonce prefix

        with self._lock:
            if request_id not in self._store:
                self._order.append(request_id)
            self._store[request_id] = blob
            self._evict_if_needed()

        logger.debug(
            "PHIVault: stored %d entities for request=%s", len(entity_map), request_id
        )

    def retrieve(self, request_id: str) -> dict[str, str]:
        """
        Decrypt and return the entity_map for *request_id*, then delete it.

        Returns an empty dict if *request_id* is not found (idempotent).

        Raises
        ------
        RuntimeError
            If decryption fails (invalid ciphertext or wrong key).
        """
        with self._lock:
            blob = self._store.pop(request_id, None)
            if request_id in self._order:
                self._order.remove(request_id)

        if blob is None:
            logger.debug("PHIVault: no entry for request=%s", request_id)
            return {}

        nonce      = blob[:_NONCE_BYTES]
        ciphertext = blob[_NONCE_BYTES:]

        try:
            plaintext = self._aesgcm.decrypt(nonce, ciphertext, associated_data=None)
        except Exception as exc:
            raise RuntimeError(
                f"PHIVault: decryption failed for request {request_id} — "
                "data may be corrupt or the vault key has changed"
            ) from exc

        entity_map: dict[str, str] = json.loads(plaintext.decode())
        logger.debug(
            "PHIVault: retrieved %d entities for request=%s", len(entity_map), request_id
        )
        return entity_map

    def delete(self, request_id: str) -> None:
        """Remove a stored entry without decrypting it."""
        with self._lock:
            self._store.pop(request_id, None)
            if request_id in self._order:
                self._order.remove(request_id)

    def __len__(self) -> int:
        """Number of entries currently held in the vault."""
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict the oldest entry when the store exceeds ``_ttl``. Lock must be held."""
        while len(self._order) > self._ttl:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)
            logger.warning("PHIVault: evicted oldest entry request=%s (ttl exceeded)", oldest)

    @staticmethod
    def generate_key_b64() -> str:
        """
        Generate a fresh random 256-bit key encoded as URL-safe base64.

        Useful for bootstrapping: run once, store the output as ``PHI_VAULT_KEY``.

        Example::

            key = PHIVault.generate_key_b64()
            # → "abc123..."  (44 characters)
            # export PHI_VAULT_KEY=abc123...
        """
        return base64.urlsafe_b64encode(secrets.token_bytes(_KEY_BYTES)).decode()
