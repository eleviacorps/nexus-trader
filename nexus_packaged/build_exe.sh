#!/bin/bash
set -e

echo "[1/5] Renaming sensitive modules for obfuscation..."
mv nexus_packaged/protection/encryptor.py nexus_packaged/protection/_e7f3.py
mv nexus_packaged/protection/integrity.py nexus_packaged/protection/_b2a1.py
sed -i 's/from nexus_packaged.protection.encryptor/from nexus_packaged.protection._e7f3/g' nexus_packaged/protection/__init__.py
sed -i 's/from nexus_packaged.protection.integrity/from nexus_packaged.protection._b2a1/g' nexus_packaged/protection/__init__.py

echo "[2/5] Running Nuitka compilation..."
python -m nuitka nexus_packaged/main.py \
    --standalone \
    --onefile \
    --lto=yes \
    --follow-imports \
    --python-flag=no_annotations \
    --python-flag=no_docstrings \
    --include-data-dir=nexus_packaged/data=nexus_packaged/data \
    --include-data-dir=nexus_packaged/config=nexus_packaged/config \
    --include-data-dir=nexus_packaged/protection=nexus_packaged/protection \
    --output-dir=nexus_packaged/dist \
    --output-filename=nexus_trader.exe \
    --assume-yes-for-downloads

echo "[3/5] Computing executable hash..."
sha256sum nexus_packaged/dist/nexus_trader.exe > nexus_packaged/dist/nexus_trader.exe.sha256
echo "Hash stored at nexus_packaged/dist/nexus_trader.exe.sha256"

echo "[4/5] Restoring original module names..."
mv nexus_packaged/protection/_e7f3.py nexus_packaged/protection/encryptor.py
mv nexus_packaged/protection/_b2a1.py nexus_packaged/protection/integrity.py
sed -i 's/from nexus_packaged.protection._e7f3/from nexus_packaged.protection.encryptor/g' nexus_packaged/protection/__init__.py
sed -i 's/from nexus_packaged.protection._b2a1/from nexus_packaged.protection.integrity/g' nexus_packaged/protection/__init__.py

echo "[5/5] Build complete. Output: nexus_packaged/dist/nexus_trader.exe"

