"""Ed25519 signing of package manifests. The private key stays in ~/.tinyml/keys, never in the repo."""

from __future__ import annotations

import base64
import hashlib

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from tinyml.config import keys_dir


def _key_path():
    return keys_dir() / "signing_ed25519.pem"


def private_key() -> Ed25519PrivateKey:
    path = _key_path()
    if path.exists():
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        assert isinstance(key, Ed25519PrivateKey)
        return key
    key = Ed25519PrivateKey.generate()
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    path.chmod(0o600)
    return key


def public_pem() -> bytes:
    return private_key().public_key().public_bytes(serialization.Encoding.PEM,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo)


def key_id() -> str:
    raw = private_key().public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def sign(data: bytes) -> str:
    return base64.b64encode(private_key().sign(data)).decode()


def verify(data: bytes, signature_b64: str, public_key_pem: bytes) -> bool:
    key = serialization.load_pem_public_key(public_key_pem)
    assert isinstance(key, Ed25519PublicKey)
    try:
        key.verify(base64.b64decode(signature_b64), data)
        return True
    except InvalidSignature:
        return False
