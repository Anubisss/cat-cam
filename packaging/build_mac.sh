#!/bin/bash
# Builds the standalone app and its disk image: dist/Cat-Cam-<version>.dmg.
# Needs macOS and uv; run from anywhere: packaging/build_mac.sh
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(uv run python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
APP="dist/Cat Cam.app"
DMG="dist/Cat-Cam-$VERSION.dmg"

echo "Drawing the icon..."
rm -rf build/AppIcon.iconset
uv run python packaging/make_icon.py build/AppIcon.iconset
iconutil -c icns build/AppIcon.iconset -o build/AppIcon.icns

echo "Building the app..."
uv run --group build pyinstaller --noconfirm --clean --log-level WARN \
    --distpath dist --workpath build/pyinstaller packaging/cat-cam.spec
codesign --force --deep --sign - "$APP"  # ad hoc: no Apple developer account

echo "Checking the app..."
"$APP/Contents/MacOS/Cat Cam" --check

echo "Making the disk image..."
STAGING=build/dmg
rm -rf "$STAGING" "$DMG"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"
cp "packaging/Read me first.txt" "$STAGING/"
hdiutil create -quiet -volname "Cat Cam" -srcfolder "$STAGING" -format UDZO "$DMG"
echo "Done: $DMG ($(du -h "$DMG" | cut -f1))"
