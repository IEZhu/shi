// C ABI over the macOS Core Audio process-tap API (macOS 14.2+).
//
// Captures the system audio mix — everything the machine is playing, which
// during a call is every remote participant — without a virtual audio device
// and without the "Screen Recording" permission ScreenCaptureKit would demand.
#ifndef SHI_TAP_H
#define SHI_TAP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct shi_tap shi_tap;

// Called on a private dispatch queue with interleaved float samples.
// Realtime context: must not allocate, lock or block.
typedef void (*shi_tap_audio_cb)(void *ctx,
                                 const float *samples,
                                 uint32_t frame_count,
                                 uint32_t channels);

// Start tapping the global system mix.
// Returns NULL on failure and writes the failing OSStatus to out_status.
shi_tap *shi_tap_start(shi_tap_audio_cb cb, void *ctx, int32_t *out_status);

// Stop and release everything. Safe to call with NULL.
void shi_tap_stop(shi_tap *tap);

uint32_t shi_tap_sample_rate(const shi_tap *tap);
uint32_t shi_tap_channels(const shi_tap *tap);

// Copy a human-readable device description as NUL-terminated UTF-8.
void shi_tap_device_name(const shi_tap *tap, char *buf, size_t buf_len);

#ifdef __cplusplus
}
#endif
#endif
