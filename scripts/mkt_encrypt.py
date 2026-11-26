"""
Verschlüsselt eine Marktupdate-Ausgabe (JSON) für das öffentliche Repository.

Aufruf:  python scripts/mkt_encrypt.py ausgabe.json
Ergebnis: manual/marktupdate/<JJJJ>-KW<NN>.json.enc

Verfahren: zufälliger AES-256-GCM-Schlüssel für den Inhalt, dieser Schlüssel mit dem öffentlichen
RSA-Schlüssel (manual/marktupdate_public.pem, OAEP/SHA-256) verschlüsselt. Entschlüsseln kann nur der
Workflow mit dem GitHub-Secret MKT_PRIVATE_KEY. Klartext landet nie im Repository.
"""
import base64
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parent.parent


def main(src: str):
    ed = json.loads(Path(src).read_text(encoding="utf-8"))
    for k in ("jahr", "kw", "datum", "sections"):
        if k not in ed:
            sys.exit(f"Pflichtfeld fehlt: {k}")
    pub = serialization.load_pem_public_key((ROOT / "manual" / "marktupdate_public.pem").read_bytes())
    key, iv = AESGCM.generate_key(256), os.urandom(12)
    ct = AESGCM(key).encrypt(iv, json.dumps(ed, ensure_ascii=False).encode("utf-8"), None)
    ek = pub.encrypt(key, padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    b = lambda x: base64.b64encode(x).decode()  # noqa: E731
    out = ROOT / "manual" / "marktupdate" / f"{ed['jahr']}-KW{int(ed['kw']):02d}.json.enc"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"v": 1, "alg": "RSA-OAEP-256+AES-256-GCM", "ek": b(ek), "iv": b(iv), "data": b(ct)}))
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main(sys.argv[1])
