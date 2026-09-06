# Direct display

Status: **unsupported / not evaluated** (Phases S–T). The experimental
Linux/NVIDIA direct-output prototype (Vulkan `VK_KHR_display` +
exportable-memory + CUDA external-memory import + external semaphores +
DRM/KMS timing) is feature-gated and will emit UNSUPPORTED/INCONCLUSIVE
receipts rather than fabricated passes when hardware/API constraints prevent
meaningful testing.

Non-goals (per prompt §32–34): assuming Vulkan image == physical scanout
allocation; calling shared-VRAM "zero copy"; calling a full-frame allocation
a VOLE frame; claiming photon latency from software clocks.
