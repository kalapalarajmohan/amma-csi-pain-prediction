# amma-csi-pain-prediction

Contactless WiFi CSI sensing for nocturnal movement and (eventually) pre-onset pain guarding — built to monitor my mother's chronic sciatic pain without a camera or wearable in her room.

## What this is

An ESP32 streams WiFi Channel State Information (CSI) over UDP to a Jetson Nano, which runs a live edge pipeline (`amma_40nights.py`) as a self-healing systemd service. The system reliably detects gross body movement and repositioning through the night — no camera, no wearable, $30 in hardware.

## What this is not (yet)

The end goal is detecting **pre-onset micro-muscular guarding** — the reflexive muscle tensing that may precede a conscious pain episode — not just gross motion. Right now, this is a motion detector, not a pain predictor. That distinction matters and I'd rather be upfront about it than oversell what's here.

## Why CSI, not RSSI

The first version of this system used plain WiFi **RSSI** (a single signal-strength scalar read once per second via `iwconfig`), thresholded against a fixed baseline to detect spikes. It didn't work for this purpose — RSSI collapses all spatial and multipath information into one number, so it could register that *something* changed near the router but couldn't distinguish a person shifting in bed from someone walking through the next room. It was giving shadows, not shapes.

Switching to **CSI** (52 subcarriers of amplitude and phase per packet from the ESP32) gave per-subcarrier resolution instead of one blended number — enough to start resolving gross body movement and repositioning, which is what the current pipeline runs on.

## The core open problem

Gross repositioning produces amplitude changes clearly visible above ambient noise in this setup. The fine-grained muscular guarding I actually want to catch is, by rough estimate, several orders of magnitude smaller — likely well below this system's current noise floor. I have not yet formally characterized that noise floor or measured this gap precisely; doing so is the next concrete step, along with exploring better sensor placement, a different modality, or additional signal processing.

## Honest findings and failure modes

- **Night 1 observation:** a high-band subcarrier ratio spike appeared roughly 1 minute before a reported pain episode. This is *not* confirmed as predictive. It's equally consistent with a reactive postural shift that occurred after pain onset but before it was reported. I don't yet have a way to distinguish the two, and I'm treating this as an open question rather than a result.
- **Posture classification is broken on real hardware.** A kernel SVM classifier that scored 86% in simulation cycles incoherently between upright/slouched/lying labels in deployment — the subcarrier ratios for these postures are nearly identical given this room's fixed geometry. Simulation accuracy did not transfer.
- **Breathing rate calculation was silently wrong** for a period due to a hardcoded sample rate (100 Hz) that didn't match the actual bursty ~26–34 packets/sec from the ESP32. Fixed by computing sample rate from real window timestamps.
- **A dead data path ran unnoticed for months** — an old collection script was listening on a port the ESP32 had stopped using, since May.

## Status

- 8 of 40 target nights collected as of writing; data collection is the current bottleneck, not modeling.
- Two live dashboards: a raw subcarrier heatmap and a plain-language status view (CALM / MOVEMENT / SOME ACTIVITY) for non-technical use.
- Not pursuing invasive ground-truth sensing (e.g. surface EMG) at this time — my mother's comfort takes priority over data collection, and I'd rather report an honest limitation than push past what's appropriate for the person this project is meant to help.
- Independently validated the breathing-rate extraction against real recorded data: a second, separately-written implementation of the rate-detection logic agreed with the live pipeline's own output on 26 of 30 sampled windows within 5 bpm, run on the actual deployment hardware.

## Why this repo is public in this state

I'd rather show real hardware, real deployment nights, and real failure analysis than a polished simulation. If you're a researcher working on ambient physiological sensing, contactless health monitoring, or the gap between what commodity RF hardware can resolve and what biological signals actually require — I'd like to hear from you.

Rajmohan Kalapala
