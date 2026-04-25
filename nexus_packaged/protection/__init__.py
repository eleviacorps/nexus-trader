"""Protection primitives for packaged runtime."""

from __future__ import annotations

from nexus_packaged.protection.encryptor import (
    decrypt_model_to_buffer,
    derive_key_from_env,
    encrypt_model_weights,
)
from nexus_packaged.protection.integrity import compute_exe_hash, verify_integrity

__all__ = [
    "derive_key_from_env",
    "encrypt_model_weights",
    "decrypt_model_to_buffer",
    "compute_exe_hash",
    "verify_integrity",
]




















































































































