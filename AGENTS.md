# AGENTS.md — Tracker Failure Simulator

## Project Purpose & Scope
This project is an **interactive failure and stress-testing simulator for Multi-Object Tracking (MOT) and Single-Object Tracking algorithms**.

> [!IMPORTANT]
> **Local-Only Execution**: This project is built strictly as a locally runnable research, development, and diagnostic tool. It is **not** intended, designed, or licensed for hosted SaaS deployment, multi-tenant cloud services, or commercial product offerings.

### Key Architecture Principles
1. **Simulation-Only Focus**:
   - There are **no deep learning object detection models** (e.g. weights like `.pt` files) or real-world video upload pipelines.
   - The simulator generates synthetic, ground-truth trajectories mathematically (crossing paths, occlusion walls, camera jitter, motion blur, appearance look-alikes).
   - Upstream detector imperfection is modeled synthetically via `SimDetector` (configurable false alarm rates, missed detection probabilities, and bounding box jitter).

2. **Pure Tracker Benchmarking**:
   - The simulator tests how tracking association logic (Kalman filtering, Hungarian matching, IoU association, ReID appearance embeddings, BYTE thresholding) performs under controlled conditions.
   - All trackers evaluated side-by-side in a simulation run receive the **exact same detection stream**, ensuring fair, reproducible MOT benchmarks (MOTA, MOTP, IDF1, ID switches, fragments, and latency).

3. **Supported Trackers**:
   - **Standalone MOT Engines**: `bytetrack`, `botsort`, `ocsort`, `deepocsort`, `fasttrack`, `tracktrack` (implemented natively in standalone pure NumPy/SciPy without external deep learning frameworks or `torch`).
   - **Custom Baselines**: `greedy_iou`, `centroid`, `sort`, `embed_sort` (implemented from scratch in pure NumPy/SciPy without external downloads).
   - **OpenCV Single-Object Trackers**: `kcf`, `csrt`, `mosse`, `mil`, `medianflow`, `nano`, `vit`, `dasiamrpn`.

4. **API Surface**:
   - `GET /api/health`: Health status.
   - `GET /api/trackers`: Catalog of available trackers, scenario detector parameters, and default configurations.
   - `GET /api/defaults`: Default detector & scenario noise parameters.
   - `POST /api/scenarios/preview`: Render and stream preview of synthetic scene with ground-truth overlays.
   - `POST /api/simulations`: Start asynchronous multi-tracker evaluation job.
   - `GET /api/jobs/{id}`: Poll status, events, natural-language failure diagnostics, and metrics.
   - `POST /api/jobs/{id}/cancel`: Request cancellation of a queued or running job; triggers cancellation tokens across worker threads and marks job status as `cancelled`.
   - `GET /api/media/{file}`: Retrieve rendered video clips (WebM/MP4) and failure thumbnails.
   - `GET /api/logs/files`: List available server log files.
   - `GET /api/logs`: Paginated log reader with cursor-based pagination.
   - `POST/DELETE /api/clear` (or `/api/cleanup`): Purge generated artifacts (`backend/data/jobs/**/*.mp4`, `backend/data/jobs/**/*.jpg`, `backend/data/scenarios/*.mp4`); protected with `X-Admin-Key` header when `SIM_ADMIN_KEY` is set.

5. **Client Lifecycle & Tab State Isolation**:
   - **Per-Tab Job Isolation**: Active job requests are persisted with a unique client tab identifier in browser storage (`sessionStorage` and `localStorage`). If User A starts Job 1 and User B starts Job 2 concurrently in different tabs, each tab retains and tracks its own distinct job ID without state collision.
   - **Background Tab Persistence**: Navigating away from the tab or switching between workspace views (Standard, Production, Logs) retains the mounted simulator state and polling. Returning to the tab re-synchronizes live job status via `visibilitychange`.
   - **Refresh & Cancellation Protection**: If a diagnostic simulation is running and the user reloads or navigates away, a `beforeunload` dialog warns that the running worker will be stopped. Upon confirmation, a `sendBeacon` / `keepalive` cancellation token is fired to abort worker threads and free resources.

6. **Contributor Guidelines**:
   - **Pure NumPy/SciPy Implementations**: New trackers must be implemented natively or integrated via standalone pure Python/NumPy logic. Do not introduce dependencies on `ultralytics`, `torch`, or download pretrained weights (`.pt`/`.onnx`).
   - **Local Research Scope**: All functionality must run locally without SaaS hosting dependencies or cloud telemetry.
   - **Tests**: Always accompany new tracker algorithms or scenario mechanics with unit tests in `backend/tests/` that pass with `pytest`.
