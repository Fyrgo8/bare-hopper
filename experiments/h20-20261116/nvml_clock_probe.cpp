// Isolated GPU 0 probe. Compile against the installed NVML header and driver.
// Does not reset GPUs, stop services, or write deferred settings.
#include <nvml.h>
#include <cstdio>

void report(char const *operation, nvmlReturn_t status) {
  std::printf("%s: code=%d, message=%s\n", operation, int(status),
              nvmlErrorString(status));
}

int main() {
  auto status = nvmlInit_v2();
  report("nvmlInit_v2", status);
  if (status != NVML_SUCCESS) return 1;
  nvmlDevice_t device{};
  status = nvmlDeviceGetHandleByUUID(
      "GPU-d1b595ce-0b57-cdee-c1c0-9b1051aaa07e", &device);
  report("nvmlDeviceGetHandleByUUID", status);
  if (status != NVML_SUCCESS) { nvmlShutdown(); return 1; }
  status = nvmlDeviceSetMemoryLockedClocks(device, 1593, 1593);
  report("nvmlDeviceSetMemoryLockedClocks(1593,1593)", status);
  // If supported, always restore the supported original memory target.
  if (status == NVML_SUCCESS)
    report("nvmlDeviceSetMemoryLockedClocks(2619,2619)",
           nvmlDeviceSetMemoryLockedClocks(device, 2619, 2619));
  nvmlEnableState_t current{}, default_state{};
  report("nvmlDeviceGetAutoBoostedClocksEnabled",
         nvmlDeviceGetAutoBoostedClocksEnabled(device, &current, &default_state));
  report("nvmlShutdown", nvmlShutdown());
}
