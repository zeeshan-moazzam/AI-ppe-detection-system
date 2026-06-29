# AI-ppe-detection-system
<div align="center">

<img src="assets/logo.png" alt="PPE Detection Logo" width="180"/>

# AI-Powered Industrial PPE Detection & Safety Surveillance System

**Real-time Personal Protective Equipment (PPE) violation detection using YOLOv8, multi-camera RTSP streams, and an automated 4-process alerting pipeline.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-Web%20Dashboard-green?logo=flask)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-RTSP%20Streaming-red)](https://opencv.org)
[![SQLite](https://img.shields.io/badge/SQLite-Database-lightblue)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-Academic%20%26%20Research-yellow)](LICENSE)

*Deployed in a live industrial environment at **PT Indo Bharat Rayon**, monitoring 9+ IP cameras across a manufacturing facility for real-time PPE compliance enforcement.*

</div>

---

## Live Detection Samples

<div align="center">

| Camera 1 | Camera 2 | Camera 3 |
|:---:|:---:|:---:|
| ![Cam 1](assets/detection_cam1.jpg) | ![Cam 2](assets/detection_cam2.jpg) | ![Cam 3](assets/detection_cam3.jpg) |

| Camera 4 | Camera 5 | Camera 6 |
|:---:|:---:|:---:|
| ![Cam 4](assets/detection_cam4.jpg) | ![Cam 5](assets/detection_cam5.jpg) | ![Cam 6](assets/detection_cam6.jpg) |

| Camera 7 | Camera 8 | Camera 9 |
|:---:|:---:|:---:|
| ![Cam 7](assets/detection_cam7.jpg) | ![Cam 8](assets/detection_cam8.jpg) | ![Cam 9](assets/detection_cam9.jpg) |

</div>

---

## Violation Video Evidence

> The system automatically records an MP4 clip whenever a sustained PPE violation (3+ consecutive frames) is confirmed. Below is a sample clip.

https://github.com/user-attachments/assets/sample_violation.mp4

> To embed your actual violation video on GitHub, upload `assets/sample_violation.mp4` as a GitHub release asset and replace the link above with the raw URL from your repository's releases page.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Process Pipeline](#process-pipeline)
5. [Dashboard & UI](#dashboard--ui)
6. [Violation Detection & Video Evidence](#violation-detection--video-evidence-1)
7. [Database Schema](#database-schema)
8. [Installation & Setup](#installation--setup)
9. [Configuration](#configuration)
10. [Running the System](#running-the-system)
11. [API Reference](#api-reference)
12. [User Management & Roles](#user-management--roles)
13. [Email Alerts & Reports](#email-alerts--reports)
14. [Tuning Parameters](#tuning-parameters)
15. [Data Directories & Retention](#data-directories--retention)
16. [Project Structure](#project-structure)
17. [Dependencies](#dependencies)
18. [Deployment Notes](#deployment-notes)
19. [License](#license)

---

## Overview

This system continuously monitors industrial workers via IP/RTSP cameras and uses a **custom-trained YOLOv8** model (`best.pt`) to detect PPE violations in real time. When a violation is confirmed (3+ consecutive frames), the system:

- Records an **MP4 video clip** of the violation event
- Saves the **best snapshot frame**
- Writes the event to a **SQLite database**
- Sends **email alerts** to configured recipients
- Displays everything on a **live web dashboard**

The system is built around a **loosely coupled 4-process pipeline** connected via CSV log files and a shared SQLite database, making each process independently restartable without data loss.

---

## Key Features

| Feature | Details |
|---|---|
| Live camera feeds | Up to 9+ simultaneous RTSP IP cameras via MJPEG streams |
| PPE violation detection | Helmet, vest, gloves, mask, eyewear — configurable per camera |
| Violation streak logic | 3 consecutive frames required (eliminates single-frame false positives) |
| Video evidence | MP4 clips auto-created for every confirmed violation |
| Snapshot evidence | Best-frame annotated JPEG saved per violation |
| Real-time dashboard | Flask web UI with live feeds, stats, charts |
| Role-based access | Admin (all cameras) vs User (assigned cameras only) |
| Email alerts | SMTP alerts with violation type, camera, and timestamp |
| Automated PDF/email reports | Scheduled reports with embedded matplotlib charts |
| Camera management | Add/edit/delete cameras via UI without restarting |
| User management | Add/edit/delete users, reset passwords via UI |
| GPU acceleration | CUDA GPU batch inference with automatic CPU fallback |
| Per-camera class filters | Each camera monitors a different subset of PPE classes |
| Hot-reload config | Camera changes applied every 30s without restart |
| Data auto-cleanup | Auto-delete raw frames (2h capture, 4h detection) |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     RTSP IP Cameras (9+)                          │
│             Hikvision / generic IP cameras on LAN                 │
└────────────────────────────┬─────────────────────────────────────┘
                             │ RTSP (one connection per camera)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Process 1: app.py                             │
│  Flask Web Server  │  RTSP Stream Manager  │  REST API            │
│  • Owns ALL RTSP connections (prevents double-connect)            │
│  • Serves /api/frame/<cam_id> → latest JPEG                      │
│  • Dashboard UI (login, live feeds, charts, violations)           │
│  • Camera / user / email CRUD operations                          │
│  • Queries ppe.db and serves data via REST API                   │
│  • Sends email alerts and generates PDF reports                   │
│  • http://localhost:5000                                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP GET /api/frame/<cam_id> @ 1 FPS
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Process 2: capture.py                           │
│  Frame Capture Worker (one thread per camera)                     │
│  • Polls app.py for the latest frame per camera                   │
│  • Saves JPEGs → captured_frames/<cam_id>/<date>/<hour>/         │
│  • Appends metadata → logs/capture.csv                            │
│  • Auto-deletes frames older than 2 hours                         │
└────────────────────────────┬─────────────────────────────────────┘
                             │ logs/capture.csv (IPC via CSV tail)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Process 3: detect.py                            │
│  YOLOv8 Batch Inference Engine                                    │
│  • CaptureReader: tails capture.csv via byte-offset cursor        │
│  • Pushes new rows into per-camera queues                         │
│  • BatchInferenceEngine: drains up to 16 frames per GPU call     │
│  • Draws bounding boxes on annotated frames                       │
│  • Saves detected_frames/ and violation_snapshots/                │
│  • Appends results → logs/detection.csv                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │ logs/detection.csv (IPC via CSV tail)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Process 4: monitor.py                           │
│  Violation Streak Tracker                                         │
│  • Polls detection.csv every 5 seconds                            │
│  • Tracks consecutive violation frames per camera/type            │
│  • FRAMES_THRESHOLD=3 → triggers confirmed violation              │
│  • GAP_TOLERANCE=5 → handles brief occlusions                    │
│  • COOLDOWN_MINUTES=2 → prevents duplicate alerts                 │
│  • Creates MP4 clips using FFmpeg                                 │
│  • Inserts records into ppe.db (SQLite)                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │ ppe.db (SQLite, WAL mode)
                             ▼
                  app.py dashboard reads ppe.db
          (violation history, charts, PDF reports)
```

**Why CSV-based IPC?**
Each process communicates via append-only CSV files read with a byte-offset cursor (like `tail -f`). This makes every process independently restartable — if `detect.py` crashes, it resumes from where it left off, and the upstream `capture.py` keeps writing without blocking.

**Why does `app.py` own RTSP connections?**
Most IP cameras only allow one RTSP connection at a time. By routing all frame access through `app.py`'s `/api/frame/<cam_id>`, exactly one RTSP socket exists per camera at all times.

---

## Process Pipeline

### Process 1 — `app.py` (Flask Server)

The central hub. Must start first — all other processes depend on it.

**Responsibilities:**
- Opens and maintains one RTSP connection per enabled camera using OpenCV (`cv2.VideoCapture`)
- Serves the latest frame for each camera via `GET /api/frame/<cam_id>` (JPEG)
- Provides full MJPEG streaming via `GET /video_feed/<cam_id>`
- Hosts the web dashboard at `http://localhost:5000`
- Handles session-based authentication (login/logout, password reset via email token)
- Manages camera CRUD (`cameras.json`) and user CRUD (`users.json`)
- Generates matplotlib charts and sends email alerts / PDF reports
- Runs a background auto-report thread for scheduled email delivery

**Key constants:**
```python
CACHE_TTL     = 5       # seconds — summary cache TTL
ONLINE_TIMEOUT = 300    # seconds — user online threshold
TOKEN_EXPIRY   = 3600   # seconds — password reset token validity
INDONESIA_TZ   = timezone(timedelta(hours=7))  # UTC+7 WIB
```

---

### Process 2 — `capture.py` (Frame Capture Worker)

Spawns one thread per enabled camera. Each thread:
1. Polls `/api/health` until `app.py` is ready
2. Every 1 second: `GET /api/frame/<cam_id>` → saves JPEG to disk
3. Appends a row to `logs/capture.csv`
4. Every 10 minutes: deletes frames older than 2 hours

**CSV output (`logs/capture.csv`):**
```
timestamp,frame_location,camera_ip,camera_id
2026-05-14 19:00:00,captured_frames/cam1/20260514/19/frame_20260514_190000.jpg,rtsp://...,cam1
```

**Key constants:**
```python
APP_BASE_URL  = "http://127.0.0.1:5000"
FPS           = 1     # frames per second per camera
CLEANUP_HOURS = 2     # delete frames older than this
```

---

### Process 3 — `detect.py` (YOLOv8 Batch Inference Engine)

Designed to saturate the GPU instead of serialising one frame at a time:

**`CaptureReader` thread:**
- Tails `logs/capture.csv` via a persistent byte-offset cursor
- Parses each new row and pushes it into a per-camera `queue.Queue`

**`BatchInferenceEngine` thread:**
- Every 200ms, drains up to `BATCH_SIZE=16` frames (round-robin across cameras)
- Calls `model(batch_frames)` under `model_lock` (GPU-exclusive section)
- Releases the lock immediately after inference; all post-processing runs in parallel:
  - Per-camera class filtering (applied post-inference — batch can't use per-image masks)
  - `is_violation()` check on class names containing `'no-'`, `'without'`, `'no_'`, `'missing'`
  - Draws bounding boxes + labels + timestamp watermark
  - Saves annotated frame to `detected_frames/<cam_id>/`
  - Saves best snapshot to `violation_snapshots/<cam_id>/<YYYYMMDD>/`
  - Appends result row to `logs/detection.csv`

**CSV output (`logs/detection.csv`):**
```
captured_timestamp,processed_timestamp,has_violation,violation_types,
violation_count,num_persons,processed_frame_location,snapshot_location,
original_frame,camera_id
```

**Key constants:**
```python
CONF_THRESHOLD  = 0.35   # YOLO confidence cutoff
BATCH_SIZE      = 16     # frames per GPU inference call
BATCH_INTERVAL  = 0.20   # seconds — batch collection window
CLEANUP_HOURS   = 4      # delete annotated frames after this
USE_GPU         = True   # auto-falls back to CPU
```

---

### Process 4 — `monitor.py` (Violation Streak Tracker)

Polls `logs/detection.csv` every 5 seconds. Maintains per-camera, per-violation-type state:

| State variable | Purpose |
|---|---|
| `start` | Timestamp of first frame in current streak |
| `frames` | List of frame paths collected in streak (capped at 120) |
| `snap` | Best snapshot path (highest simultaneous violation count) |
| `count` | Consecutive violation frames so far |
| `saved` | Whether this streak already triggered an alert |
| `last_alert` | Timestamp of last triggered alert (for cooldown check) |
| `gap` | Consecutive clear frames seen since last violation |

**Streak trigger sequence:**
1. Frame with `has_violation=true` → `count++`
2. If `count >= 3` and not in 2-minute cooldown → **trigger violation event**
   - Call FFmpeg to encode MP4 from collected frame list
   - Save record to `ppe.db`
   - Set `saved=True`, record `last_alert` timestamp
3. Frame with `has_violation=false`:
   - If `gap < 5` → `gap++` (streak survives brief occlusion)
   - If `gap >= 5` → fully reset streak state for that type

**Key constants:**
```python
FFMPEG_PATH       = r"C:\ffmpeg\bin\ffmpeg.exe"  # update per machine
FRAMES_THRESHOLD  = 3   # consecutive frames before triggering
COOLDOWN_MINUTES  = 2   # minimum gap before re-alerting same type
GAP_TOLERANCE     = 5   # clear frames allowed before streak resets
MIN_FRAMES_FOR_VIDEO = 3
```

---

## Dashboard & UI

The web dashboard is served at `http://localhost:5000`.

### Login Page (`/login`)

- Username + password form with password visibility toggle
- Uses werkzeug `check_password_hash()` with scrypt hashing
- Session-based with `login_required` and `admin_required` decorators
- Password reset via emailed token (1-hour expiry)

### Admin Dashboard (`/`)

**Live Camera Grid:**
- MJPEG streams for all enabled cameras
- Camera name, Online/Offline indicator, last-seen timestamp
- Click to open full-screen camera view

**Statistics Cards:**
- Total violations (all time), Violations today, Active cameras, Compliance %

**Charts:**
- Hourly violations histogram (last 24 hours)
- Violation type breakdown (donut pie chart)
- Camera-wise violation count (horizontal bar chart)

**Violation History Table:**
- Filterable by: date range, violation type, camera, status
- Columns: Type, Camera, Start time, End time, Duration, Frame count, Actions
- Actions: Watch MP4 video (in-browser player), view snapshot, delete record

**Admin Panel:**
- Camera management (add/edit/delete, RTSP URL, class filters)
- User management (add/edit/delete, role, camera access)
- Email config (SMTP host, port, credentials)
- Report config (recipients, schedule, interval)

### User Dashboard (`/user`)

- Shows only cameras assigned to that user
- Live feed grid (restricted view)
- Recent violations for assigned cameras only
- No admin panel or management controls

### Violations Page (`/violations`)

- Multi-filter toolbar (date range, type multi-select, camera multi-select)
- Sortable, paginated table with summary statistics
- Bulk delete (admin only), export to PDF or CSV

---

## Violation Detection & Video Evidence

Every confirmed violation (3+ consecutive frames) generates an MP4 clip:

**Video creation process:**
1. `monitor.py` collects the list of annotated frame paths during the streak (up to 120 frames)
2. Writes a temporary file-list for FFmpeg
3. Calls: `ffmpeg -r 2 -f concat -safe 0 -i frames.txt -c:v libx264 -preset ultrafast output.mp4`
4. Output saved to `violation_videos/<cam_id>_<type>_<timestamp>.mp4`

**Video specs:**
- Frame rate: 2 FPS (1 FPS captures played at 2× speed)
- Codec: H.264 (libx264), ultrafast preset
- Each frame is the annotated (bounding-box drawn) version

**Snapshot selection logic:**
- Frame with the highest simultaneous `violation_count` is selected as the "best" snapshot
- Bounding box labels: `violation_type + confidence score` (e.g., `no-helmet 0.87`)
- Timestamp watermark in bottom-right corner

---

## Database Schema

**File:** `ppe.db` (SQLite, WAL mode)

```sql
CREATE TABLE violations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT,        -- e.g. "cam1"
    viol_type   TEXT,        -- e.g. "no-helmet", "no-vest|no-gloves"
    start_time  TEXT,        -- "YYYY-MM-DD HH:MM:SS" (WIB)
    end_time    TEXT,        -- "YYYY-MM-DD HH:MM:SS" (WIB)
    video_path  TEXT,        -- relative path to MP4 file
    snap_path   TEXT,        -- relative path to snapshot JPEG
    frame_count INTEGER,     -- number of frames in the streak
    created_at  TEXT         -- insertion timestamp (WIB)
);

CREATE INDEX idx_v_start   ON violations(start_time);
CREATE INDEX idx_v_camera  ON violations(camera_id);
CREATE INDEX idx_v_type    ON violations(viol_type);
CREATE INDEX idx_v_created ON violations(created_at);
```

**Pragmas at every connection:**
```sql
PRAGMA journal_mode=WAL;    -- concurrent readers during writes
PRAGMA busy_timeout=5000;   -- wait up to 5s on lock before error
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- FFmpeg (required for violation video creation)
- CUDA-compatible GPU (optional — CPU fallback is automatic)
- Access to RTSP camera streams on your network

### Step 1 — Clone the Repository

```bash
git clone https://github.com/zeeshan-moazzam/AI-PPE-Detection.git
cd AI-PPE-Detection
```

### Step 2 — Create Virtual Environment

```bash
python -m venv myvenv

# Windows
myvenv\Scripts\activate

# Linux / macOS
source myvenv/bin/activate
```

### Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Optional (for charts and PDF reports):**
```bash
pip install matplotlib reportlab
```

### Step 4 — Install FFmpeg

**Windows:**
1. Download from [ffmpeg.org](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg\`
3. Update `FFMPEG_PATH` in `monitor.py` line 28:
   ```python
   FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
   ```

**Linux:**
```bash
sudo apt install ffmpeg
```
Then set `FFMPEG_PATH = "ffmpeg"` in `monitor.py`.

### Step 5 — Place Your YOLOv8 Model

Copy your trained model to the project root:
```
best.pt
```
It must be a standard Ultralytics YOLOv8 `.pt` file.

---

## Configuration

### `cameras.json` — Camera Registry

```bash
cp cameras.json.example cameras.json
```

```json
{
  "cam1": {
    "name": "Main Entrance",
    "rtsp_url": "rtsp://admin:password@192.168.1.100/Streaming/Channels/101",
    "enabled": true,
    "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  },
  "cam2": {
    "name": "Production Floor",
    "rtsp_url": "rtsp://admin:password@192.168.1.101/Streaming/Channels/101",
    "enabled": true,
    "classes": [0, 5, 6]
  }
}
```

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name shown in dashboard |
| `rtsp_url` | string | Full RTSP URL with credentials |
| `enabled` | boolean | If false, camera is ignored by all processes |
| `classes` | int array | YOLO class indices to monitor (per-camera subset) |

> Cameras can be added, removed, or toggled while running — `detect.py` hot-reloads `cameras.json` every 30 seconds.

---

### `users.json` — User Accounts

```bash
cp users.json.example users.json
```

```json
{
  "admin": {
    "name": "Administrator",
    "email": "admin@company.com",
    "password_hash": "scrypt:32768:8:1$...",
    "role": "admin",
    "cameras": []
  },
  "operator1": {
    "name": "Operator One",
    "email": "operator1@company.com",
    "password_hash": "scrypt:32768:8:1$...",
    "role": "user",
    "cameras": ["cam1", "cam3"]
  }
}
```

| Field | Description |
|---|---|
| `role` | `"admin"` = full access, `"user"` = restricted to assigned cameras |
| `cameras` | Camera IDs this user can view (`[]` for admin = all cameras) |

---

### `email_config.json` — SMTP Alert Settings

```bash
cp email_config.json.example email_config.json
```

```json
{
  "smtp_host": "smtp.gmail.com",
  "smtp_port": 587,
  "smtp_user": "alerts@company.com",
  "smtp_password": "your_app_password",
  "from_email": "alerts@company.com"
}
```

Supported ports: `25` (plain relay), `587` (STARTTLS), `465` (SMTP_SSL).

---

### `report_config.json` — Scheduled Report Settings

```json
{
  "recipients": [
    {
      "email": "manager@company.com",
      "label": "Safety Manager",
      "auto_send": true,
      "interval_hours": 1.0,
      "last_sent": "2026-05-14 10:00:00"
    }
  ]
}
```

Each recipient can have an independent send interval. The auto-report thread in `app.py` checks every minute and sends when `now - last_sent >= interval_hours`.

> **Never commit `cameras.json`, `email_config.json`, or `users.json` with real credentials.** These files are listed in `.gitignore`.

---

## Running the System

### Option A — Automatic (Recommended, Windows)

```bat
start_all.bat
```

Launches all 4 processes in the correct order in separate terminal windows.

To stop everything:
```bat
stop_all.bat
```

### Option B — Silent Background Mode (Windows)

```bat
cscript start_silent.vbs
```

Runs all processes invisibly. Logs are written to `logs/<process>.log`.

### Option C — Manual (Cross-platform)

Open **4 separate terminals** and run in this exact order:

```bash
# Terminal 1 — start first, all others depend on it
python app.py

# Terminal 2 — after app.py is ready
python capture.py

# Terminal 3
python detect.py

# Terminal 4
python monitor.py
```

Then open: **`http://localhost:5000`**

---

## API Reference

All endpoints require session authentication unless noted.

### System

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/health` | None | Health check — returns `{"status":"ok"}` |
| `GET` | `/api/summary` | User | Cached violation summary (5s TTL) |

### Camera Management

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/cameras` | User | List all cameras with status |
| `POST` | `/api/cameras` | Admin | Add new camera |
| `PUT` | `/api/cameras/<cam_id>` | Admin | Update camera config |
| `DELETE` | `/api/cameras/<cam_id>` | Admin | Delete camera |
| `GET` | `/api/frame/<cam_id>` | None | Latest JPEG frame |
| `GET` | `/video_feed/<cam_id>` | User | MJPEG stream |

### Violations

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/violations` | User | Query violations (filter: date, camera, type) |
| `GET` | `/api/recent_violations` | User | Last N violations (`?n=10`) |
| `GET` | `/api/violation_stats` | User | Aggregated stats by type/camera/hour |
| `GET` | `/snapshot/<path>` | User | Serve snapshot JPEG |
| `GET` | `/video/<path>` | User | Serve MP4 (supports Range headers) |

### User Management

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/users` | Admin | List all users |
| `POST` | `/api/users` | Admin | Create user |
| `PUT` | `/api/users/<username>` | Admin | Update user |
| `DELETE` | `/api/users/<username>` | Admin | Delete user |
| `POST` | `/api/reset_password_request` | None | Send password reset email |

### Reports

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/report_config` | Admin | Get report schedule |
| `POST` | `/api/report_config` | Admin | Update report schedule |
| `POST` | `/api/send_report` | Admin | Manually trigger report email |
| `GET` | `/api/generate_pdf` | Admin | Download PDF report |

---

## User Management & Roles

| Capability | Admin | User |
|---|:---:|:---:|
| View all cameras | Yes | Assigned only |
| Live camera feed | Yes | Assigned only |
| View violations | Yes | Assigned cameras only |
| Delete violations | Yes | No |
| Export violations (PDF/CSV) | Yes | No |
| Manage cameras | Yes | No |
| Manage users | Yes | No |
| Configure email/reports | Yes | No |
| Send reports manually | Yes | No |
| Reset own password | Yes | Yes |

---

## Email Alerts & Reports

### Violation Alerts

Triggered automatically when a violation is confirmed. Alert email includes:
- Camera name and ID
- Violation type(s)
- Start and end timestamp
- Duration and frame count
- Attached snapshot image

### Scheduled Reports

- **Interval:** hourly, daily, or custom (`interval_hours`)
- **Format:** HTML email with embedded charts + optional PDF attachment
- **Charts included:** Hourly histogram, violation type pie, camera bar chart
- **PDF report:** Generated with ReportLab, includes same charts

Configure via the Admin Panel → Report Configuration, or directly in `report_config.json`.

---

## Tuning Parameters

| Constant | File | Default | Effect |
|---|---|---|---|
| `FPS` | `capture.py` | `1` | Capture rate per camera — increase for faster detection, more storage |
| `CLEANUP_HOURS` | `capture.py` | `2` | Hours before raw frames are auto-deleted |
| `CONF_THRESHOLD` | `detect.py` | `0.35` | YOLO confidence cutoff — raise to reduce false positives |
| `BATCH_SIZE` | `detect.py` | `16` | Frames per GPU call — reduce if VRAM is low |
| `BATCH_INTERVAL` | `detect.py` | `0.20s` | Batch collection window |
| `CLEANUP_HOURS` | `detect.py` | `4` | Hours before annotated frames are auto-deleted |
| `FRAMES_THRESHOLD` | `monitor.py` | `3` | Consecutive violation frames before alert fires |
| `COOLDOWN_MINUTES` | `monitor.py` | `2` | Minimum gap between alerts per camera/type |
| `GAP_TOLERANCE` | `monitor.py` | `5` | Clear frames allowed before resetting a streak |
| `CACHE_TTL` | `app.py` | `5` | Summary API cache lifetime (seconds) |

---

## Data Directories & Retention

| Directory | Contents | Retention |
|---|---|---|
| `captured_frames/<cam>/<date>/<hour>/` | Raw JPEG frames | 2 hours (auto-deleted) |
| `detected_frames/<cam>/` | Annotated frames with bounding boxes | 4 hours (auto-deleted) |
| `violation_snapshots/<cam>/<date>/` | Best-frame JPEGs per violation | Permanent |
| `violation_videos/` | MP4 clips of violation events | Permanent |
| `logs/capture.csv` | Frame capture log (IPC) | Permanent (append-only) |
| `logs/detection.csv` | Detection results log (IPC) | Permanent (append-only) |
| `ppe.db` | SQLite violation records | Permanent |

**Manual data reset:**
```bash
python clear_violations.py
# Prompts for confirmation "YES" before deleting all violation data
```

---

## Project Structure

```
AI-PPE-Detection/
│
├── Core Processes
│   ├── app.py                  Flask server: dashboard, REST API, RTSP manager, alerts
│   ├── capture.py              Frame capture worker (1 FPS per camera)
│   ├── detect.py               YOLOv8 batch inference engine
│   ├── monitor.py              Violation streak tracker + MP4 creator
│   ├── clear_violations.py     Data reset utility (requires "YES" confirmation)
│   └── reset_data.py           Full system reset utility
│
├── Configuration
│   ├── cameras.json            Active camera registry (create from .example)
│   ├── cameras.json.example    Camera config template
│   ├── users.json              User accounts with hashed passwords (create from .example)
│   ├── users.json.example      User config template
│   ├── email_config.json       SMTP settings (create from .example)
│   ├── email_config.json.example
│   ├── report_config.json      Scheduled report settings (create from .example)
│   └── report_config.json.example
│
├── ML Model
│   ├── best.pt                 Custom-trained YOLOv8 model (not included — add your own)
│   └── yolov8n.pt              Pretrained YOLOv8 nano (reference)
│
├── Web Templates (Jinja2)
│   └── templates/
│       ├── login.html
│       ├── dashboard.html      Admin dashboard
│       ├── user_dashboard.html Restricted user view
│       ├── violations.html     Violation history browser
│       └── reset_password.html Password reset page
│
├── Assets
│   ├── assets/logo.png
│   ├── assets/detection_cam1.jpg ... detection_cam9.jpg
│   └── assets/sample_violation.mp4
│
├── Database
│   └── ppe.db                  SQLite database (auto-created on first run)
│
├── Logs (auto-created)
│   └── logs/
│       ├── capture.csv         IPC: capture → detect
│       ├── detection.csv       IPC: detect → monitor
│       ├── app.log
│       ├── capture.log
│       ├── detect.log
│       └── monitor.log
│
├── Frame Storage (auto-created, auto-cleaned)
│   ├── captured_frames/        Raw JPEGs
│   ├── detected_frames/        Annotated JPEGs with bounding boxes
│   ├── violation_snapshots/    Best snapshot per violation event
│   └── violation_videos/       MP4 clips
│
├── Startup Scripts (Windows)
│   ├── start_all.bat           Start all 4 processes (visible windows)
│   ├── stop_all.bat            Kill all processes
│   └── start_silent.vbs        Start silently (no terminal windows)
│
└── requirements.txt
```

---

## Dependencies

### Python Packages

| Package | Purpose |
|---|---|
| `Flask` | Web framework — dashboard and REST API |
| `opencv-python` | RTSP stream management, frame I/O, annotation |
| `ultralytics` | YOLOv8 model loading and inference |
| `numpy` | Array operations for frame batching |
| `torch` / `torchvision` | PyTorch backend for YOLOv8 |
| `werkzeug` | Password hashing (bundled with Flask) |
| `matplotlib` *(optional)* | Charts for reports |
| `reportlab` *(optional)* | PDF report generation |

### System Dependencies

| Tool | Purpose | Required |
|---|---|---|
| FFmpeg | MP4 video encoding for violation clips | **Yes** |
| CUDA Toolkit | GPU acceleration for YOLO inference | No (CPU fallback) |

---

## Deployment Notes

### Timezone

All timestamps use **Indonesia WIB (UTC+7)**. If deploying in a different timezone, update in all three files (`app.py`, `detect.py`, `monitor.py`):
```python
INDONESIA_TZ = timezone(timedelta(hours=7))
```

### FFmpeg Path

Hardcoded in `monitor.py` line 28 — update this before first run on any new machine.

### Flask Secret Key

Auto-generated and persisted in `.flask_secret` on first run. In `.gitignore` — never commit it.

### Production Checklist

- [ ] Update `cameras.json` with real RTSP URLs
- [ ] Set strong passwords in `users.json`
- [ ] Configure `email_config.json` with your SMTP relay
- [ ] Update `FFMPEG_PATH` in `monitor.py` to your FFmpeg binary path
- [ ] Verify GPU is detected — check `detect.py` startup logs for CUDA device
- [ ] Confirm `ppe.db` is writable by the process user
- [ ] Set up disk monitoring — `violation_videos/` and `violation_snapshots/` grow unbounded

### Hot-Reload Behavior

- Adding a new camera: picked up within 30 seconds, no restart needed
- Removing a camera: thread stopped and queue removed within 30 seconds
- Changing RTSP URLs: requires restarting `app.py` (it holds the RTSP connection)

---

## License

This project is licensed for **Academic and Research Use** only.

Commercial use, redistribution, and modification for production deployment require explicit written permission from the project author.

See [LICENSE](LICENSE) for full terms.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection framework
- [OpenCV](https://opencv.org) — Computer vision library
- [Flask](https://flask.palletsprojects.com) — Web framework
- [FFmpeg](https://ffmpeg.org) — Video encoding
- [ReportLab](https://www.reportlab.com) — PDF generation
- [Matplotlib](https://matplotlib.org) — Chart generation
- **PT Indo Bharat Rayon** — Production deployment and real-world testing environment

---

<div align="center">

**Author: Zeeshan Moazzam**

zeeshanmoazzam22@gmail.com

*Deployed and tested in a live industrial environment at PT Indo Bharat Rayon, monitoring 9 IP cameras across a manufacturing facility for real-time PPE compliance enforcement.*

</div>

