# ClipAgent — Product Vision & Mission

## What We're Building

A **local-first, AI-powered long-form video → short-form clip pipeline** that runs on our own hardware and agent infrastructure. The goal is feature parity with tools like Opus Clip, built on our stack, with no per-minute fees, no API rate limits from third-party clip services, and full control over every stage of the pipeline.

We own the 3090. We own the agent layer. We build the tool.

---

## Mission Statement

> Turn any long-form video into platform-ready short clips — automatically, intelligently, and without leaving our own infrastructure.

The system should find the best moments, score them for virality, reframe them for vertical formats, burn in styled captions, and be ready to push to any platform — with as little human intervention as the content allows.

---

## The Six Stages

### Stage 1 — Ingest & Transcription
- Accept local file, URL, or YouTube link as input
- Run ASR (Whisper or equivalent) to produce a full transcript with **per-word timestamps**
- Optional: speaker diarization for multi-person content
- Output: timestamped transcript JSON, ready for LLM consumption

### Stage 2 — LLM Understanding & Segmentation
- Feed transcript to our agent (Claude via agent API) with structured prompts to:
  - Segment the video into logical chapters / topics
  - Score each segment on: hook strength, value density, narrative clarity, self-containedness
  - Optionally filter by user-specified topic, keyword, or tone
  - Identify "gold nugget" moments — high-value fragments that can be stitched into a coherent non-contiguous clip
- Output: ranked list of clip candidates with start/end timestamps and metadata

### Stage 3 — Virality Scoring
- Per-clip scalar score from 0–100 used to rank and prioritize clips
- Signal inputs:
  - Hook phrasing and opening energy
  - Sentiment and emotional arc
  - Pacing and speech density
  - Topic relevance vs. current trend tags
  - Speaking clarity and confidence
  - Structural patterns: setup → payoff length, tension/release
- This can start as a weighted LLM scoring pass and evolve toward a trained model as data accumulates

### Stage 4 — Visual Reframing
- Convert 16:9 source to 9:16, 1:1, or custom AR
- Subject/face tracking with saliency-aware crop window (the `reframe/` pipeline)
- Genre-aware layout logic:
  - Talking head: face-centered crop, InsightFace precision
  - Gameplay + facecam: split layout or facecam priority toggle
  - Screenshare + speaker: content region top, speaker pip bottom
  - Cars/tools/objects: YOLO class-weighted saliency, no face anchor needed
- Smooth camera path via Kalman + shot-mode classifier (STATIONARY / PAN / TRACK)
- AutoFlip-style edge case handling: occlusion decay, split-subject fallback, edge bias

### Stage 5 — Rendering
- FFmpeg-based compositor (GPU-accelerated where possible on the 3090)
- Bake captions with per-word timing, word-level highlights, animated pop-ins
- Emoji overlays keyed to sentiment/keywords
- Cut and stitch non-contiguous segments with clean transitions
- Output: platform-ready MP4, H.264/AAC, correct SAR, burned-in style layer

### Stage 6 — Distribution
- Direct publish or scheduled upload to: YouTube Shorts, TikTok, Instagram Reels
- Metadata generation: AI-written title, description, hashtags per platform
- Local archive of all output clips with virality scores and metadata

---

## Guiding Principles

**Local-first.** The 3090 does the heavy lifting. No per-clip billing. No rate limits from third-party services. We run Whisper locally, we run YOLO locally, we run the reframe pipeline locally.

**Agent-native.** The LLM understanding layer runs through our own agent API (Claude). Prompts are versioned, logged, and improvable. The agent isn't a black box — it's part of our stack.

**Full pipeline ownership.** Every stage is ours. We can inspect, modify, and improve any layer. No inherited assumptions from tools we don't control.

**Benchmarkable output.** Every clip has a virality score, a source timestamp, and a decision log. We can review why a clip was selected and improve the scoring over time.

**Modular by design.** Each stage (ingest, segment, score, reframe, render, distribute) is independent. You can swap the ASR model, tune the scoring prompt, or replace the reframe engine without touching the others.

---

## Stack Summary

| Stage | Component | Runs On |
|---|---|---|
| Ingest / ASR | Whisper (large-v3 or turbo) | Local / 3090 |
| Segmentation & scoring | Claude via agent API | API |
| Virality score | LLM scoring → future trained model | API / Local |
| Reframing | `reframe/` pipeline — YOLO + InsightFace + Kalman | Local / 3090 |
| Caption rendering | FFmpeg + custom burn layer | Local |
| Distribution | Platform APIs (YouTube, TikTok, IG) | Network |

---

## What We're Not Doing (right now)

- No multi-user SaaS, no team seats, no billing infrastructure
- No cloud rendering — the 3090 is the render node
- No proprietary caption animation framework — FFmpeg burn-in first, upgrade later
- No mobile app

---

## First Milestone

A working end-to-end run on a single long-form video:

1. Whisper transcript with word timestamps
2. Claude segments + scores the transcript, returns top 5 clip candidates
3. `reframe/` pipeline converts each clip to 9:16
4. Captions burned in via FFmpeg
5. Output folder with 5 ranked clips ready to review

Everything after that is iteration.
