# Optional Android QNN HTP policy

Ordinary builds retain the existing profiles, QNN runtime, provider options and
cache names. Three independent build-time options are available:

| Gradle property | Effect |
| --- | --- |
| `jetlink.qnnSharedIO=true` | Allocate host tensors through ORT's actual QNN HTP shared allocator and bind them with the matching memory information. |
| `jetlink.qnnStrict=true` | Disable CPU execution-provider fallback in HTP sessions during both compilation and reload. Unsupported graphs fail instead of silently leaving compute on CPU. |
| `jetlink.qnnNoActivationFusion=true` | Use the optional matching-SDK adapter to disable advanced HTP activation fusion, a numerical workaround. |

All default to false. The policy is typed; JSON configuration accepts Booleans,
not strings or numeric substitutes. GPU and CPU sessions do not register their
bindings as HTP shared memory. These policy options leave original models and
preparation unchanged; the optional causal profile is documented separately in
[android-causal-prefill.md](android-causal-prefill.md).

For shared I/O and strict HTP admission using the normal runtime:

```bash
cd android
./gradlew :app:assembleRelease :app:testDebugUnitTest \
  -Pjetlink.qnnSharedIO=true -Pjetlink.qnnStrict=true
```

The optional adapter requires locally obtained headers and libraries from one
matching Qualcomm SDK. Vendor headers, libraries and models are not included or
downloaded. Use the project's normal Android/Swift build prerequisites, then:

```bash
export QNN_SDK_ROOT=/path/to/one/qairt-sdk
export ANDROID_NDK_HOME=/path/to/android-ndk
bash android/qnn/build.sh /path/to/NEW-qnn-jni
cd android
./gradlew :app:assembleRelease :app:testDebugUnitTest \
  -Pjetlink.qnnSharedIO=true -Pjetlink.qnnStrict=true \
  -Pjetlink.qnnNoActivationFusion=true -Pjetlink.qnnJniDir=/path/to/NEW-qnn-jni
```

The helper runs a host forwarding/lifetime test, builds the adapter and stages
Android libraries plus available matched HTP stub/skel pairs from that same SDK.
The SDK must expose `QNN_HTP_GRAPH_CONFIG_OPTION_ADVANCED_ACTIVATION_FUSION`.
The adapter forwards the public backend interface and disables only this option.
It retains custom configuration through successful `contextFree`, including when
the backend consumes configuration after graph creation. API incompatibility,
missing libraries, missing allocators or unsupported graphs produce errors.

Gradle replaces the transitive Maven QNN runtime only in this explicit adapter
build, avoiding a mixture of runtimes. It fingerprints the actual staged library
bytes for the cache namespace; these are not fixed vendor-binary or SoC pins.
The runtime directory and identity flow through `QnnPolicy`, without global
tuning variables. Each changed policy has a different HTP cache namespace.

## Evidence and limits

On one Xiaomi 15 / Snapdragon 8 Elite, a historical whole-NPU CTV3 test using
ORT 1.29 and QAIRT 2.47 compared the same range-safe prepared graph with stock
advanced activation fusion versus fusion disabled. The stock capture failed
**six output slices** despite finite values; the disabled-fusion capture passed
all original gates across **32 identical input frames**. This establishes a
numerical workaround for that tested stack, not a universal cause or speedup.
The preparation used in that experiment is not added or changed here.

A separate sequential one-minute comparison of the same whole-NPU engine
reported ordinary memory **42.9 ms mean / 45.2 ms P99**, versus shared I/O
**40.5 ms mean / 42.2 ms P99**. The mean reduction was about 5.6%. Order,
temperature and clocks were not fully controlled, so this is indicative evidence,
not a general performance guarantee. Strict fallback prevention is a correctness
contract rather than a speed optimization. None of these measurements establishes
50% improvement, 100% NPU occupancy or vehicle readiness.

Before promoting an option on another graph/runtime/device, retain original
precision and use the unchanged original-model parity gates on identical inputs
with independent recurrent state, including reset tests. Compare alternating
heap/shared runs under similar temperatures, then perform sustained and cold-load
tests. Record exact runtime/options, package/native identities and mean/P99/max.
The current patch's host tests do not qualify an HTP device or a car connection.
