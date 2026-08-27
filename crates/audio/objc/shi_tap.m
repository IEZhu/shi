#import "shi_tap.h"

#import <AudioToolbox/AudioToolbox.h>
#import <CoreAudio/CoreAudio.h>
// The CoreAudio umbrella header pulls in only AudioHardware.h and
// HostTime.h; the tap API lives in its own ObjC-only headers.
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#import <Foundation/Foundation.h>

struct shi_tap {
    AudioObjectID tap_id;
    AudioObjectID agg_id;
    AudioDeviceIOProcID proc_id;
    shi_tap_audio_cb cb;
    void *ctx;
    uint32_t sample_rate;
    uint32_t channels;
    char device_name[256];
};

#pragma mark - property helpers

static OSStatus shi_get_prop(AudioObjectID object,
                             AudioObjectPropertySelector selector,
                             AudioObjectPropertyScope scope,
                             UInt32 *io_size,
                             void *out_data) {
    AudioObjectPropertyAddress address = {
        .mSelector = selector,
        .mScope = scope,
        .mElement = kAudioObjectPropertyElementMain,
    };
    return AudioObjectGetPropertyData(object, &address, 0, NULL, io_size, out_data);
}

/// UID of the current default output device. The aggregate uses it as its
/// clock source so the tap does not drift against the hardware.
static NSString *shi_default_output_uid(NSString **out_name) {
    AudioObjectID device = kAudioObjectUnknown;
    UInt32 size = sizeof(device);
    if (shi_get_prop(kAudioObjectSystemObject,
                     kAudioHardwarePropertyDefaultOutputDevice,
                     kAudioObjectPropertyScopeGlobal,
                     &size, &device) != noErr || device == kAudioObjectUnknown) {
        return nil;
    }

    CFStringRef uid = NULL;
    size = sizeof(uid);
    if (shi_get_prop(device, kAudioDevicePropertyDeviceUID,
                     kAudioObjectPropertyScopeGlobal, &size, &uid) != noErr) {
        return nil;
    }

    if (out_name) {
        CFStringRef name = NULL;
        UInt32 name_size = sizeof(name);
        if (shi_get_prop(device, kAudioObjectPropertyName,
                         kAudioObjectPropertyScopeGlobal, &name_size, &name) == noErr && name) {
            *out_name = (__bridge_transfer NSString *)name;
        }
    }

    return (__bridge_transfer NSString *)uid;
}

#pragma mark - lifecycle

shi_tap *shi_tap_start(shi_tap_audio_cb cb, void *ctx, int32_t *out_status) {
    if (out_status) {
        *out_status = noErr;
    }
    if (cb == NULL) {
        if (out_status) *out_status = kAudio_ParamError;
        return NULL;
    }

    @autoreleasepool {
        // A mono global tap is exactly what the pipeline wants: the whole
        // system mix, already downmixed, at the device's native rate.
        CATapDescription *description =
            [[CATapDescription alloc] initMonoGlobalTapButExcludeProcesses:@[]];
        description.name = @"Shi Meeting Capture";
        // Private: visible only to us, so we never leave a stray device behind
        // in the user's Audio MIDI Setup.
        description.privateTap = YES;
        // Unmuted: the user must still hear the meeting they are in.
        description.muteBehavior = CATapUnmuted;

        AudioObjectID tap_id = kAudioObjectUnknown;
        OSStatus status = AudioHardwareCreateProcessTap(description, &tap_id);
        if (status != noErr || tap_id == kAudioObjectUnknown) {
            if (out_status) *out_status = status != noErr ? status : kAudioHardwareUnspecifiedError;
            return NULL;
        }

        CFStringRef tap_uid_ref = NULL;
        UInt32 size = sizeof(tap_uid_ref);
        status = shi_get_prop(tap_id, kAudioTapPropertyUID,
                              kAudioObjectPropertyScopeGlobal, &size, &tap_uid_ref);
        if (status != noErr || tap_uid_ref == NULL) {
            AudioHardwareDestroyProcessTap(tap_id);
            if (out_status) *out_status = status != noErr ? status : kAudioHardwareUnspecifiedError;
            return NULL;
        }
        NSString *tap_uid = (__bridge_transfer NSString *)tap_uid_ref;

        AudioStreamBasicDescription asbd = {0};
        size = sizeof(asbd);
        status = shi_get_prop(tap_id, kAudioTapPropertyFormat,
                              kAudioObjectPropertyScopeGlobal, &size, &asbd);
        if (status != noErr) {
            AudioHardwareDestroyProcessTap(tap_id);
            if (out_status) *out_status = status;
            return NULL;
        }

        NSString *output_name = nil;
        NSString *output_uid = shi_default_output_uid(&output_name);

        NSMutableDictionary *aggregate = [@{
            (__bridge NSString *)CFSTR(kAudioAggregateDeviceNameKey): @"Shi Capture",
            (__bridge NSString *)CFSTR(kAudioAggregateDeviceUIDKey): [[NSUUID UUID] UUIDString],
            (__bridge NSString *)CFSTR(kAudioAggregateDeviceIsPrivateKey): @YES,
            (__bridge NSString *)CFSTR(kAudioAggregateDeviceIsStackedKey): @NO,
            (__bridge NSString *)CFSTR(kAudioAggregateDeviceTapAutoStartKey): @YES,
            (__bridge NSString *)CFSTR(kAudioAggregateDeviceSubDeviceListKey): @[],
            (__bridge NSString *)CFSTR(kAudioAggregateDeviceTapListKey): @[@{
                (__bridge NSString *)CFSTR(kAudioSubTapUIDKey): tap_uid,
                (__bridge NSString *)CFSTR(kAudioSubTapDriftCompensationKey): @YES,
            }],
        } mutableCopy];

        if (output_uid) {
            aggregate[(__bridge NSString *)CFSTR(kAudioAggregateDeviceMainSubDeviceKey)] = output_uid;
        }

        AudioObjectID agg_id = kAudioObjectUnknown;
        status = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)aggregate, &agg_id);
        if (status != noErr || agg_id == kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tap_id);
            if (out_status) *out_status = status != noErr ? status : kAudioHardwareUnspecifiedError;
            return NULL;
        }

        shi_tap *tap = calloc(1, sizeof(shi_tap));
        if (tap == NULL) {
            AudioHardwareDestroyAggregateDevice(agg_id);
            AudioHardwareDestroyProcessTap(tap_id);
            if (out_status) *out_status = kAudio_MemFullError;
            return NULL;
        }

        tap->tap_id = tap_id;
        tap->agg_id = agg_id;
        tap->cb = cb;
        tap->ctx = ctx;
        tap->sample_rate = (uint32_t)asbd.mSampleRate;
        tap->channels = asbd.mChannelsPerFrame ? asbd.mChannelsPerFrame : 1;
        snprintf(tap->device_name, sizeof(tap->device_name), "System output (%s)",
                 output_name ? [output_name UTF8String] : "default");

        shi_tap_audio_cb callback = cb;
        void *callback_ctx = ctx;
        uint32_t fallback_channels = tap->channels;

        status = AudioDeviceCreateIOProcIDWithBlock(
            &tap->proc_id, agg_id, /* dispatch queue */ NULL,
            ^(const AudioTimeStamp *inNow,
              const AudioBufferList *inInputData,
              const AudioTimeStamp *inInputTime,
              AudioBufferList *outOutputData,
              const AudioTimeStamp *inOutputTime) {
                (void)inNow; (void)inInputTime; (void)outOutputData; (void)inOutputTime;
                if (inInputData == NULL || inInputData->mNumberBuffers == 0) {
                    return;
                }
                // A mono mixdown tap presents one interleaved buffer.
                const AudioBuffer *buffer = &inInputData->mBuffers[0];
                if (buffer->mData == NULL || buffer->mDataByteSize == 0) {
                    return;
                }
                uint32_t channels = buffer->mNumberChannels ? buffer->mNumberChannels
                                                            : fallback_channels;
                uint32_t frames = buffer->mDataByteSize / (uint32_t)sizeof(float) / channels;
                if (frames == 0) {
                    return;
                }
                callback(callback_ctx, (const float *)buffer->mData, frames, channels);
            });

        if (status != noErr) {
            AudioHardwareDestroyAggregateDevice(agg_id);
            AudioHardwareDestroyProcessTap(tap_id);
            free(tap);
            if (out_status) *out_status = status;
            return NULL;
        }

        status = AudioDeviceStart(agg_id, tap->proc_id);
        if (status != noErr) {
            AudioDeviceDestroyIOProcID(agg_id, tap->proc_id);
            AudioHardwareDestroyAggregateDevice(agg_id);
            AudioHardwareDestroyProcessTap(tap_id);
            free(tap);
            if (out_status) *out_status = status;
            return NULL;
        }

        return tap;
    }
}

void shi_tap_stop(shi_tap *tap) {
    if (tap == NULL) {
        return;
    }
    // Unwind in reverse. Leaking an aggregate device would leave a phantom
    // entry in the user's Audio MIDI Setup that survives the process.
    if (tap->proc_id != NULL) {
        AudioDeviceStop(tap->agg_id, tap->proc_id);
        AudioDeviceDestroyIOProcID(tap->agg_id, tap->proc_id);
    }
    if (tap->agg_id != kAudioObjectUnknown) {
        AudioHardwareDestroyAggregateDevice(tap->agg_id);
    }
    if (tap->tap_id != kAudioObjectUnknown) {
        AudioHardwareDestroyProcessTap(tap->tap_id);
    }
    free(tap);
}

uint32_t shi_tap_sample_rate(const shi_tap *tap) {
    return tap ? tap->sample_rate : 0;
}

uint32_t shi_tap_channels(const shi_tap *tap) {
    return tap ? tap->channels : 0;
}

void shi_tap_device_name(const shi_tap *tap, char *buf, size_t buf_len) {
    if (buf == NULL || buf_len == 0) {
        return;
    }
    if (tap == NULL) {
        buf[0] = '\0';
        return;
    }
    snprintf(buf, buf_len, "%s", tap->device_name);
}
