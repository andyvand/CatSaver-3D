#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

BUNDLE_NAME="CatScreensaver"
SAVER="${BUNDLE_NAME}.saver"
EXEC="${BUNDLE_NAME}"

rm -rf "$SAVER"
mkdir -p "$SAVER/Contents/MacOS"
mkdir -p "$SAVER/Contents/Resources"

cp Info.plist "$SAVER/Contents/Info.plist"

SRC=(
    Sources/CatScreensaverView.swift
    Sources/Math.swift
    Sources/Mesh.swift
    Sources/CatScene.swift
    Sources/Renderer.swift
)
BRIDGING_HEADER="Sources/ShaderTypes.h"
FRAMEWORKS=(
    -framework ScreenSaver
    -framework Metal
    -framework MetalKit
    -framework QuartzCore
    -framework AppKit
    -framework Foundation
)

# Metal 4 needs macOS 26.
TARGET_OS="macos26.0"

# ---- Ship the .metal source into the bundle's Resources --------------------
# We compile shaders at runtime via the Metal 4 compiler (MTL4Compiler.
# makeLibrary(descriptor:) with a source string). This avoids needing the
# offline Metal toolchain (xcodebuild -downloadComponent MetalToolchain).

echo "→ copying Metal shader source"
cp Sources/Shaders.metal "$SAVER/Contents/Resources/Shaders.metal"

# ---- Compile per-arch slices and lipo into a universal bundle binary -------

build_slice () {
    local arch="$1"
    local target="${arch}-apple-${TARGET_OS}"
    local out="$SAVER/Contents/MacOS/${EXEC}-${arch}"
    echo "→ compiling slice: ${arch}"
    xcrun -sdk macosx swiftc \
        -O \
        -emit-library \
        -Xlinker -bundle \
        -module-name "$EXEC" \
        -target "$target" \
        -import-objc-header "$BRIDGING_HEADER" \
        -o "$out" \
        "${FRAMEWORKS[@]}" \
        "${SRC[@]}"
}

build_slice arm64
build_slice x86_64

echo "→ lipo into universal binary"
lipo -create -output "$SAVER/Contents/MacOS/$EXEC" \
    "$SAVER/Contents/MacOS/${EXEC}-arm64" \
    "$SAVER/Contents/MacOS/${EXEC}-x86_64"
rm "$SAVER/Contents/MacOS/${EXEC}-arm64" "$SAVER/Contents/MacOS/${EXEC}-x86_64"

echo "→ ad-hoc codesign"
codesign --force --deep --sign - "$SAVER"

echo ""
echo "Built: $SAVER"
echo ""
echo "To install for the current user:"
echo "  cp -R \"$SAVER\" ~/Library/Screen\\ Savers/"
echo "Then open System Settings → Screen Saver and pick 'CAT3D'."
echo ""
echo "Quick preview (no install):"
echo "  /System/Library/Frameworks/ScreenSaver.framework/Resources/ScreenSaverEngine.app/Contents/MacOS/ScreenSaverEngine -module CatScreensaver"
