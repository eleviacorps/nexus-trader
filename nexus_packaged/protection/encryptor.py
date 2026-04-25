"""Model-at-rest encryption utilities.

This module implements AES-256-GCM encryption/decryption and PBKDF2 key
derivation from an environment variable. Decrypted model bytes are only
returned as in-memory buffers.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def derive_key_from_env(*, env_var: str, salt: str, iterations: int = 390000) -> bytes:
    """Derive a 32-byte AES key from an environment variable.

    Args:
        env_var: Environment variable name containing the user secret.
        salt: Fixed salt value from configuration.
        iterations: PBKDF2 iteration count.

    Returns:
        32-byte key suitable for AES-256-GCM.

    Raises:
        RuntimeError: If the secret environment variable is missing.
    """
    secret = os.getenv(env_var, "")
    if not secret:
        raise RuntimeError(
            f"{env_var} is not set. Set the environment variable before startup "
            "to enable model decryption."
        )
    kdf = PBKDF2HMAC(
        algorithm=SHA256(),
        length=32,
        salt=salt.encode("utf-8"),
        iterations=int(iterations),
    )
    return kdf.derive(secret.encode("utf-8"))


def encrypt_model_weights(source_path: str, output_path: str, key: bytes) -> None:
    """Encrypt raw model bytes using AES-256-GCM.

    Format on disk:
        [nonce (12 bytes)] + [tag (16 bytes)] + [ciphertext]
    """
    source = Path(source_path)
    target = Path(output_path)
    plaintext = source.read_bytes()
    nonce = os.urandom(12)
    aes = AESGCM(key)
    ciphertext_with_tag = aes.encrypt(nonce, plaintext, associated_data=None)
    tag = ciphertext_with_tag[-16:]
    ciphertext = ciphertext_with_tag[:-16]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(nonce + tag + ciphertext)


def decrypt_model_to_buffer(encrypted_path: str, key: bytes) -> BytesIO:
    """Decrypt an encrypted model blob and return in-memory bytes."""
    blob = Path(encrypted_path).read_bytes()
    if len(blob) < 28:
        raise ValueError("Encrypted model payload is too short.")
    nonce = blob[:12]
    tag = blob[12:28]
    ciphertext = blob[28:]
    aes = AESGCM(key)
    plaintext = aes.decrypt(nonce, ciphertext + tag, associated_data=None)
    return BytesIO(plaintext)


def export_key_b64(key: bytes) -> str:
    """Helper for diagnostics (not used for runtime secrets)."""
    return base64.b64encode(key).decode("ascii")

