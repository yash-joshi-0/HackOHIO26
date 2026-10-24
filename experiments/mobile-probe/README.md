# Native mobile shard probe

The physical M4 iPad passed the two-layer GLM numerical fixture: all 16 positions matched the reference logits exactly, and clearing the contexts and repeating the first batch also matched. Raw evidence and the private runtime patch are in `results/`.

This validates native in-process shard execution with Metal on iOS. It does not validate mobile Iroh, the Rust core, full-model memory capacity, or useful generation from this deliberately truncated fixture.

The private runtime adds `shard_begin` and `shard_end` to `llama_model_params`. Rebuild the library and every caller against the matching header; this changes the ABI. Each model instance has its own range, so the app needs neither environment mutation nor subprocesses. `native_smoke.cpp` exercises reference, prefix, suffix, and context reset in one process.

Build the patched runtime with CMake's Xcode generator, `CMAKE_SYSTEM_NAME=iOS`, arm64, minimum iOS 16.4, static libraries, Metal enabled, and OpenMP disabled. Build target `llama` in Release. Configure this directory with the same generator/platform and:

```sh
cmake -S experiments/mobile-probe -B mobile-ios-build -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=16.4 \
  -DMESH_LLAMA_ROOT=/path/to/patched/llama.cpp \
  -DMESH_FIXTURE=/path/to/GLM-4.7-Flash-two-layer-fixture.gguf \
  -DMESH_APP_TEAM=YOUR_TEAM
cmake --build mobile-ios-build --config Release
```

The runtime libraries must be in `build-ios-probe` beneath the patched clone. Device installation requires signing and provisioning for your own device. The app writes `Documents/native-shard-result.json` for retrieval through Xcode/device tools.

## Android native test

Pixel 8a (API 34) and Galaxy S23 SM-S911U (API 36) both passed the same 16-position test and reset with maximum logit error zero on CPU. The native executable ran through ADB; this does not establish APK lifecycle behavior or persistent mesh connectivity. Reports, logs, and build metadata are in `results/`.

```sh
cmake -S experiments/mobile-probe/android -B android-native-build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk-r28c/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
  -DMESH_LLAMA_ROOT=/path/to/patched/llama.cpp
cmake --build android-native-build --target mesh-native-smoke -j 6
adb -s DEVICE_SERIAL shell mkdir -p /data/local/tmp/meshthing-probe
adb -s DEVICE_SERIAL push android-native-build/mesh-native-smoke /data/local/tmp/meshthing-probe/
adb -s DEVICE_SERIAL push /path/to/fixture.gguf /data/local/tmp/meshthing-probe/fixture.gguf
adb -s DEVICE_SERIAL shell 'chmod 700 /data/local/tmp/meshthing-probe/mesh-native-smoke && /data/local/tmp/meshthing-probe/mesh-native-smoke /data/local/tmp/meshthing-probe/fixture.gguf /data/local/tmp/meshthing-probe/result.json'
adb -s DEVICE_SERIAL pull /data/local/tmp/meshthing-probe/result.json
```

The successful build used NDK r28c (28.2.13676358), static C++ runtime, OpenMP disabled, and ARMv8.2 dot-product/FP16 instructions. It does not establish compatibility with older ARM CPUs.

## Android Iroh diagnostic

The unchanged Rust core at `bc50aec` also cross-compiled for `aarch64-linux-android` using the NDK API-28 clang linker. Its approved-peer `bridge-serve` ran on the Pixel 8a and forwarded a real native prefix session to a Mac suffix. All 16 positions matched the Mac CPU reference with maximum logit error zero. The leader reported relay at connection and direct at stream closure. ADB staged files and launched processes; no USB port forwarding carried inference.

`android/socket_main.cpp` binds only loopback, accepts the existing `MSH1\n` preface, and runs the existing shard runner in the same process after attaching the socket to stdin/stdout. It serves one session and exits. Build target: `mesh-shard-socket`. Invocation:

```sh
mesh-shard-socket 50171 fixture.gguf 0 1 128 0 4294967296 'Hello from Android' 0
llm-cluster --key /private/path/iroh.key bridge-serve \
  --target 127.0.0.1:50171 --allow-peer LEADER_PUBLIC_ID
```

The Mac uses `bridge-connect PHONE_PUBLIC_ID --listen 127.0.0.1:50270`, then `android/iroh_trial.py --port 50270 --runner /path/to/mac/mesh-shard-runner --fixture /path/to/fixture.gguf --output /new/results/directory`. Independently verify the fixture checksum on both devices before the trial; this diagnostic socket adapter does not provide the Python adapter's model-hash metadata.

Evidence is in `results/android-iroh/`. The S23's first two network trials failed address discovery with fallback DNS timeouts; its native numerical test passed. A system DNS lookup on the S23 also failed. Both Android CLI processes logged a background panic because `ndk-context` was uninitialized. The production Android app must initialize its JNI/platform context before networking; CLI launch is not a substitute for validating that lifecycle. These tests do not establish APK packaging, foreground-service behavior, suspension recovery, or unattended availability.

Available next devices: Pixel 10 Pro, then iPhone 14. Additional devices at home: S22, S25 FE, iPhone 13, iPhone 8, and a Snapdragon X Elite ThinkPad running Fedora. These are inventory, not tested support claims. Linux ARM64 remains unvalidated; the mobile Rust/Iroh integration is tracked separately as M-26.
