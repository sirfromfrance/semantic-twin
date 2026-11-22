"""Pixel colour classes used by the map node (numpy only, no ROS).

vegetation: green or olive foliage.  fire: saturated, bright red-orange (glowing).
Thresholds are parameters of the map node; the defaults were chosen from the model textures
and have to be checked on camera frames (the node publishes a debug overlay for that).
"""
import numpy as np

VEG_HUE = (45.0, 170.0)
VEG_SAT_MIN = 0.25
VEG_VAL_MIN = 0.12
FIRE_HUE_MAX = 40.0
FIRE_HUE_MIN_WRAP = 350.0
FIRE_SAT_MIN = 0.75
FIRE_VAL_MIN = 0.45


def hsv(rgb):
    """rgb: (..., 3) values 0..255 -> hue in degrees, saturation 0..1, value 0..1."""
    c = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c.max(axis=-1), c.min(axis=-1)
    d = mx - mn
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.where(mx == r, ((g - b) / d) % 6.0,
                     np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
        s = np.where(mx > 0, d / mx, 0.0)
    h = np.where(d == 0, 0.0, h)
    return h, s, mx


def vegetation(rgb, hue=VEG_HUE, sat_min=VEG_SAT_MIN, val_min=VEG_VAL_MIN):
    h, s, v = hsv(rgb)
    return (h >= hue[0]) & (h <= hue[1]) & (s >= sat_min) & (v >= val_min)


def fire(rgb, hue_max=FIRE_HUE_MAX, sat_min=FIRE_SAT_MIN, val_min=FIRE_VAL_MIN):
    h, s, v = hsv(rgb)
    return ((h <= hue_max) | (h >= FIRE_HUE_MIN_WRAP)) & (s >= sat_min) & (v >= val_min)
