import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gps_tf import GpsTfConfig, GpsTransform, enu_to_geodetic, geodetic_to_enu  # noqa: E402

ORIGIN = (34.0689, -118.4452, 100.0)


def make(**kw):
    return GpsTransform(GpsTfConfig(origin_lat=ORIGIN[0], origin_lon=ORIGIN[1], origin_alt=ORIGIN[2], **kw))


def test_geodesy_roundtrip_and_scale():
    lla = enu_to_geodetic([100.0, 200.0, 5.0], *ORIGIN)
    assert np.allclose(geodetic_to_enu(*lla, *ORIGIN), [100.0, 200.0, 5.0], atol=1e-4)
    assert abs((lla[0] - ORIGIN[0]) * 111_000 - 200.0) < 1.5  # ~111 km per degree of latitude


def test_forward_motion_follows_heading_and_scale():
    for heading, expect in [(0, (0, 20)), (90, (20, 0)), (180, (0, -20)), (270, (-20, 0))]:
        tf = make(heading_deg=heading, scale=2.0, start_height_m=1.5)
        e, n, u = tf.v_to_enu([0, 0, 10])  # 10 VO units straight ahead of the level first camera
        assert np.allclose([e, n], expect, atol=1e-6), heading
        assert abs(u - 1.5) < 1e-9


def test_right_is_clockwise_of_heading():
    tf = make(heading_deg=0, scale=1.0)
    assert np.allclose(tf.v_to_enu([5, 0, 0])[:2], [5, 0], atol=1e-9)  # camera right of north-facing = east
    assert np.allclose(make(heading_deg=90).v_to_enu([5, 0, 0])[:2], [0, -5], atol=1e-9)  # right of east = south


def test_down_in_camera_is_down_in_enu():
    tf = make(scale=1.0, start_height_m=10.0)
    assert abs(tf.v_to_enu([0, 3, 0])[2] - 7.0) < 1e-9  # +y is down, so 3 units lower


def test_pitched_camera_levels_the_frame():
    tf = make(cam_pitch_deg=30, scale=1.0, heading_deg=0, start_height_m=0.0)
    ahead = np.array([0, 0, 1.0])  # along the optical axis, 30 deg below horizontal
    e, n, u = tf.v_to_enu(ahead)
    assert abs(n - math.cos(math.radians(30))) < 1e-9 and abs(u + math.sin(math.radians(30))) < 1e-9


def test_annotate_detection_and_glasses_relative_bearing():
    tf = make(heading_deg=0, scale=1.0, start_height_m=1.5)
    tf.set_glasses(heading_true_deg=90.0)
    geo = tf.annotate([0, 0, 0, 0, 0, 0, 1], [{"id": 3, "label": "person", "center": [0, 0, 20.0]}])
    d = geo["detections"][0]
    assert d["range_m"] == 20.0 and d["bearing_true_deg"] == 0.0
    assert d["bearing_rel_deg"] == -90.0  # target is 90 deg to the left of east-facing glasses
    assert geo["origin"]["source"] == "placeholder"
    assert geo["transform_v_to_e"]["scale_status"] == "configured"
    assert abs(geo["detections"][0]["lla"][0] - ORIGIN[0]) > 1e-4  # moved north


def test_unknown_scale_is_flagged_and_has_no_sigma():
    geo = make().annotate([0, 0, 0, 0, 0, 0, 1], [])
    assert geo["transform_v_to_e"]["scale_status"] == "assumed"
    assert geo["drone"]["sigma_m"] is None


def test_set_origin_replaces_placeholder():
    tf = make()
    tf.set_origin(34.1, -118.3, 50.0)
    g = tf.annotate([0, 0, 0, 0, 0, 0, 1], [])
    assert g["origin"]["source"] == "gps" and g["origin"]["lat"] == 34.1
    assert abs(g["drone"]["lla"][0] - 34.1) < 1e-6


def test_ground_plane_autocal_recovers_scale_and_gravity():
    rng = np.random.default_rng(1)
    s_true, h = 0.25, 2.0  # true metres per VO unit; camera 2 m above ground
    pitch = math.radians(20)
    up = np.array([0, -math.cos(pitch), -math.sin(pitch)])  # true up in V
    cam = np.zeros(3)
    ground_d = up @ cam - h / s_true  # plane n.x = d
    a = np.cross(up, [1, 0, 0]); a /= np.linalg.norm(a)
    b = np.cross(up, a)
    pts = (a * rng.uniform(-20, 20, (400, 1)) + b * rng.uniform(2, 30, (400, 1)) + up * ground_d
           + up * rng.normal(0, 0.005, (400, 1)))
    pts = np.vstack([pts, rng.uniform(-10, 10, (80, 3)) + [0, 0, 15]])  # clutter
    tf = make(cam_pitch_deg=15, start_height_m=h, autocal=True)  # prior pitch off by 5 deg
    for _ in range(6):
        geo = tf.annotate([0, 0, 0, 0, 0, 0, 1], [], pcl_xyz=pts)
    t = geo["transform_v_to_e"]
    assert t["scale_status"] == "ground_plane" and t["level_status"] == "ground_plane"
    assert abs(t["scale_m_per_vo_unit"] - s_true) / s_true < 0.05
    assert abs(geo["drone"]["height_agl_m"] - h) < 0.1
    assert np.allclose(tf._up_v, up, atol=0.02)


def test_camera_block_axes_point_the_right_way():
    tf = make(heading_deg=90, scale=1.0, start_height_m=10.0)  # level camera facing east
    cam = tf.annotate([0, 0, 0, 0, 0, 0, 1], [], intrinsics=(500, 500, 320, 180), image_size=(640, 360))["camera"]
    right, down, fwd = (np.array(a) for a in cam["axes_enu"])
    assert np.allclose(fwd, [1, 0, 0], atol=1e-6)  # forward = east
    assert np.allclose(down, [0, 0, -1], atol=1e-6)  # image-down = ENU down
    assert np.allclose(right, [0, -1, 0], atol=1e-6)  # right of east-facing = south
    assert cam["enu"][2] == 10.0 and cam["image_size"] == [640, 360]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
