# llama.cpp Build Fix: ggml_build_forward_select

## Problem

Build failed with:
```
error: 'ggml_build_forward_select' was not declared in this scope
```

## Cause

- Root `CMakeLists.txt` added **asr** (whisper) before **mt** (llama.cpp).
- Whisper's ggml was built first and created the global target `ggml`.
- llama.cpp checks `if (NOT TARGET ggml)` and, when it's already present, **reuses** that target instead of building its own ggml.
- Whisper's ggml does not define `ggml_build_forward_select` (only `ggml_build_forward_expand`), so llama.cpp's code failed to compile.

## Fix

**Build order was changed** so that **mt** is built before **asr** in the root `CMakeLists.txt`:

1. llama.cpp is built first and adds its own ggml (which includes `ggml_build_forward_select`).
2. When whisper is built, it sees that `ggml` already exists and uses that target.
3. Both llama and whisper then use the same ggml from llama.cpp, which has the full API.

## What You Need to Do

**Clean and rebuild:**

```bash
cd /home/harini/armm
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

If you use the desktop build script instead:

```bash
./scripts/build_desktop.sh
```

(Ensure the desktop script uses the same root CMakeLists or the same order.)

## If You Still See Errors

- Ensure `mt` is listed before `asr` in the root `CMakeLists.txt` under "Add subdirectories".
- If you maintain a separate build (e.g. only desktop or only Android), apply the same subdirectory order there so that the library that provides the full ggml (llama.cpp) is built first.
