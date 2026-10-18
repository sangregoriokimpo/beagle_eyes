# gps_tf — VO → ENU → lat/lon

DPVO gives a camera pose with arbitrary scale and no gravity/compass, and the DJI Mini 5 Pro
altitude/GPS telemetry is locked. This package places the drone and its 3D person detections on
the map relative to a **shared start point** (drone and glasses start together).

```
V  DPVO world (first camera frame, arbitrary scale)
     └─ scale · R(heading, level) ─▶  E  local East/North/Up metres, origin = start point (ground)
                                        └─ WGS84 around origin lat/lon ─▶ lat/lon
```

Lat/lon are always derived from ENU + origin, so replacing the placeholder origin fixes everything.

## Run

```sh
# Test at a known coordinate, camera facing east, level, scale unknown (flagged "assumed")
python scripts/uav_vo.py --detect-people --gps-tf --gps-origin 34.0689,-118.4452 --gps-heading 90

# Metric scale + gravity from the ground plane (drone hovering 1.5 m above flat ground at start)
python scripts/uav_vo.py --detect-people --gps-tf --gps-origin 34.0689,-118.4452 \
    --gps-heading 90 --gps-cam-pitch 20 --gps-start-height 1.5 --gps-autocal

# Known scale instead (metres per VO unit)
python scripts/uav_vo.py --gps-tf --gps-origin 34.0689,-118.4452 --gps-scale 0.25
```

Without `--gps-origin` a placeholder (UCLA) is used and reported as `origin.source: "placeholder"`.
All flags can also be set under `gps_tf:` in `config/uav_config.yaml`.

| Flag | Meaning |
| --- | --- |
| `--gps-origin LAT,LON[,ALT]` | Start coordinate (source `fixed`) |
| `--gps-heading` | True bearing the first camera faces |
| `--gps-cam-pitch` | First-camera pitch below horizontal (assumed roll = 0) |
| `--gps-scale` | Metres per VO unit |
| `--gps-start-height` | Start height above ground (m) |
| `--gps-autocal` | RANSAC ground plane → scale + up vector |

## Output

Each telemetry packet gains a `geo` block (also reaches the FireSight bundle metadata):
`origin`, `transform_v_to_e` (`scale_status`: assumed | configured | ground_plane), `drone`
(`enu`, `lla`, `height_agl_m`, `heading_true_deg`), `glasses` (placeholder: at origin, heading
unknown), and `detections[]` with `enu`, `lla`, `range_m`, `bearing_true_deg`,
`bearing_rel_deg` (needs glasses heading) and `sigma_m` (null while scale is assumed).

## When the glasses get GPS

```python
tf.set_origin(lat, lon, alt, source="gps")          # first good fix replaces the placeholder
tf.set_glasses(enu=[e, n, u], heading_true_deg=h)    # live position/compass → relative bearings
```

## Limits

Monocular VO drifts in scale and yaw, so `sigma_m` is a rough `5% × distance travelled`.
Autocal assumes flat ground in view and a known start height. `heading` must be set from a compass
or landmark until glasses heading is wired in. Tests: `python tests/test_gps_tf.py`.
