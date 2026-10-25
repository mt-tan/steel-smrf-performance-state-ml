"""Piecewise-linear curve of the GUI plot through the predicted characteristic points of one hazard level. It is not an
analysis result: straight segments join the predicted points in increasing roof displacement.

Vertices: (0, 0), (delta_y, V_y), (delta_max, V_max), (delta_u, 0.8 V_max) and, when 0 < delta_t <= delta_u, the target
point (delta_t, V(delta_t)) at its position. A target point beyond delta_u stays out of the line (marker only) and gives
TARGET_BEYOND_ULTIMATE; predicted points that are not in the order 0 < delta_y < delta_max < delta_u give NOT_IN_ORDER,
and the line is still drawn in increasing displacement.
"""
from __future__ import annotations

TARGET_BEYOND_ULTIMATE = 'Target displacement exceeds the predicted ultimate displacement'
NOT_IN_ORDER = 'Predicted points are not in physical order'
LEGEND = 'Piecewise-linear curve through predicted points (not an analysis result)'


def target_on_line(v: dict) -> bool:
    """True when the target point is a vertex of the curve (0 < delta_t <= delta_u)."""
    return 0.0 < v['dt'] <= v['du']


def curve_points(v: dict) -> tuple[list, bool, list]:
    """v: predictions of one hazard level in physical units (dy, Vy, dmax, Vmax, du, dt, V_at_dt). Returns (vertices, the
    target point is a vertex, warnings); vertices are (displacement [m], base shear [kN], name) in increasing displacement."""
    pts = [(0.0, 0.0, 'origin'), (v['dy'], v['Vy'], 'yield'), (v['dmax'], v['Vmax'], 'peak'),
           (v['du'], 0.8 * v['Vmax'], 'ultimate')]
    warnings = []
    if not (0.0 < v['dy'] < v['dmax'] < v['du']):
        warnings.append(NOT_IN_ORDER)
    on_line = target_on_line(v)
    if on_line:
        pts.append((v['dt'], v['V_at_dt'], 'target'))
    elif v['dt'] > v['du']:
        warnings.append(TARGET_BEYOND_ULTIMATE)
    pts.sort(key=lambda p: p[0])          # stable: equal displacements keep the order above
    return pts, on_line, warnings
