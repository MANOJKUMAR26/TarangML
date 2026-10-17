import json
import zipfile

from tinyml.packaging import signing
from tinyml.packaging.package import verify_package


def test_sign_and_detect_tampering(tmp_path):
    manifest = json.dumps({"files": {"a.txt": __import__("hashlib").sha256(b"hello").hexdigest()}}).encode()
    pkg = tmp_path / "p.zip"
    with zipfile.ZipFile(pkg, "w") as zf:
        zf.writestr("tinyml/a.txt", b"hello")
        zf.writestr("tinyml/manifest.json", manifest)
        zf.writestr("tinyml/manifest.sig", signing.sign(manifest))
        zf.writestr("tinyml/signing_key.pub.pem", signing.public_pem())
    ok, _ = verify_package(pkg)
    assert ok
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(pkg) as src, zipfile.ZipFile(bad, "w") as dst:
        for item in src.namelist():
            dst.writestr(item, b"tampered" if item.endswith("a.txt") else src.read(item))
    ok, msg = verify_package(bad)
    assert not ok and "changed" in msg
