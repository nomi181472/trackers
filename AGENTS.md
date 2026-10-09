# AGENTS.md — Tracker Failure Simulator

## Project Purpose & Scope
This project is an **interactive failure and stress-testing simulator for Multi-Object Tracking (MOT) and Single-Object Tracking algorithms**.

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
   - `GET /api/trackers`: Catalog of available trackers and hyperparameter schemas.
   - `GET /api/defaults`: Default detector & scenario noise parameters.
   - `POST /api/scenarios/preview`: Render and stream preview of synthetic scene with ground-truth overlays.
   - `POST /api/simulations`: Start asynchronous multi-tracker evaluation job.
   - `GET /api/jobs/{id}`: Poll status, events, natural-language failure diagnostics, and metrics.
   - `GET /api/media/{file}`: Retrieve rendered video clips and failure thumbnails.
   - `POST/DELETE /api/clear` (or `/api/cleanup`): Clear and delete generated records (`backend/data/jobs/**/*.mp4`, `backend/data/jobs/**/*.jpg`, `backend/data/scenarios/*.mp4`).
   - Real video endpoints (`/api/real/upload`, `/api/real/jobs`) and detector model weights are intentionally omitted.
