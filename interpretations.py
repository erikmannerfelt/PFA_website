
import warnings
from pathlib import Path

import geopandas as gpd
import json
import numpy as np
import pandas as pd
import scipy.interpolate
import shapely.geometry
import xarray as xr

import paths

def read_interpretation_xy(filepath: Path, x_vals: np.ndarray | None = None) -> pd.DataFrame:
    """
    Read digitized interpretation from a JSON feature collection and return y-values per pixel x.

    If `x_vals` is None (default), behaves like the original: for each feature,
    it fills every integer pixel between the first and last x with interpolated y.

    If `x_vals` (np.uint16) is provided, the function samples only at those pixels
    (per feature) that lie within that feature's [xmin, xmax] span.
    Pixels outside the span are skipped (no extrapolation).

    Returns
    -------
    pd.DataFrame with the fields 'y' and 'i' and MultiIndex ['radar_key', 'user', 'kind', 'x'] where
    The field 'y' represents the y pixel, and i represents the line index.
    'x' is dtype uint16 and 'y' is dtype uint16.
    """
    # ---- load JSON ----
    text = Path(filepath).read_text()
    data = json.loads(text)

    # ---- resolve user and radar_key (JSON first, then path fallback) ----
    user = data.get("user")
    radar_key = data.get("radar_key")
    if user is None or radar_key is None:
        parts = filepath.parts
        # Expect .../<user>/<radar_key>/<file>.json
        if user is None:
            user = parts[-3]
        if radar_key is None:
            radar_key = parts[-2]

    features = data["features"]["features"]

    # ---- accumulators ----
    y_chunks: list[np.ndarray] = []
    x_chunks: list[np.ndarray] = []
    kind_chunks: list[np.ndarray] = []

    # ---- helper: compress contiguous duplicate x via mean ----
    def _compress_dupes_mean(xi: np.ndarray, yi: np.ndarray):
        # xi is non-decreasing; duplicates (if any) appear in contiguous runs
        if xi.size == 0:
            return xi, yi
        run_starts = np.r_[0, np.flatnonzero(xi[1:] != xi[:-1]) + 1]
        run_ends = np.r_[run_starts[1:], xi.size]
        x_u = xi[run_starts]
        sums = np.add.reduceat(yi, run_starts)
        counts = run_ends - run_starts
        y_u = sums / counts
        return x_u, y_u

    # Normalize/prepare query x once (if provided)
    if x_vals is not None:
        # ensure array, clamp to uint16 domain, and work in int64 for math
        x_query_sorted_unique = np.unique(
            np.clip(np.asarray(x_vals, dtype=np.uint16), 0, np.iinfo(np.uint16).max).astype(np.int64, copy=False)
        )

    for feature in features:
        coords = feature["geometry"]["coordinates"]
        if not coords:
            continue

        arr = np.asarray(coords, dtype=float)  # shape (n, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue  # skip malformed

        # sort by x
        arr = arr[np.argsort(arr[:, 0])]
        x_raw = arr[:, 0]
        y_raw = arr[:, 1]

        # sanity: x must be non-decreasing (duplicates allowed)
        if x_raw.size >= 2 and np.any(x_raw[1:] < x_raw[:-1]):
            raise ValueError("x must be non-decreasing within each feature")

        # bankers rounding for x -> integer pixels (int64 for math)
        x_int = np.rint(x_raw).astype(np.int64)

        # compress contiguous duplicates by mean
        x_u, y_u = _compress_dupes_mean(x_int, y_raw)
        if x_u.size == 0:
            continue

        # Determine which x to evaluate for this feature
        if x_vals is None:
            # Original behavior: full [x_min, x_max] integer grid
            x_eval = np.arange(x_u[0], x_u[-1] + 1, dtype=np.int64)
        else:
            # Only evaluate requested pixels that fall within this feature's span
            # (no extrapolation)
            # x_query_sorted_unique is ascending and unique so np.interp is efficient
            in_span = (x_query_sorted_unique >= x_u[0]) & (x_query_sorted_unique <= x_u[-1])
            x_eval = x_query_sorted_unique[in_span]
            if x_eval.size == 0:
                continue

        # Linear interpolation (float64), then banker-round and cast to uint16
        y_eval = np.interp(x_eval, x_u, y_u)
        y_eval = np.rint(y_eval)
        y_eval = np.clip(y_eval, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)

        # x level: store as uint16 to save memory
        x_eval_u16 = np.clip(x_eval, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)

        if "kind" in feature["properties"]:
            kind = feature["properties"]["kind"]
        else:
            kind = {
                "Glacier bed": "bed_unspecified",
                "Cold glacier bed": "bed_cold",
                "Glacier bed missing": "bed_missing",
                "Temperate ice": "temperate_ice",
            }[feature["properties"]["name"]]

        # stash
        y_chunks.append(y_eval)
        x_chunks.append(x_eval_u16)
        kind_chunks.append(np.repeat(kind, x_eval_u16.size))

    if not y_chunks:
        empty_index = pd.MultiIndex.from_arrays([[], [], [], []], names=["radar_key", "user", "kind", "x"])
        return pd.DataFrame({"y": [], "i": []}, index=empty_index, dtype="uint16")

    # ---- concatenate once ----
    y_all = np.concatenate(y_chunks)  # uint16
    i_all = np.concatenate([[np.uint16(i)] * len(vals) for i, vals in enumerate(y_chunks)])
    x_all = np.concatenate(x_chunks)  # uint16
    kind_all = np.concatenate(kind_chunks)  # object
    n_total = y_all.size

    mi = pd.MultiIndex.from_arrays(
        [
            np.repeat(radar_key, n_total),
            np.repeat(user, n_total),
            kind_all,
            x_all,
        ],
        names=["radar_key", "user", "kind", "x"],
    )

    return pd.DataFrame({"y": y_all, "i": i_all}, index=mi)

def chord_sample_xinds_part_distance(x, easting, northing, step_m=5.0, jump_threshold=100.0, eps=1e-12):
    """
    Find sampling points such that successive samples are step_m apart in Euclidean (chord) distance.
    Jumps (segment length > jump_threshold) split the polyline into parts. Each new part restarts sampling and
    adds only step_m to the running distance (mimicking your previous 'cap big diffs to step size').

    Inputs
    ------
    x : array-like (float/int)
        Pixel x coordinate per vertex (not assumed monotonic).
    easting, northing : array-like (float)
        Coordinates per vertex.
    step_m : float
        Desired straight-line spacing between successive samples along the polyline.
    jump_threshold : float
        Segment length above this is treated as a discontinuity (new part).
    eps : float
        Numerical tolerance.

    Returns
    -------
    x_inds : np.ndarray (uint16)
        Rounded pixel x locations for sampled points (stable-unique).
    part_idx : np.ndarray (int32)
        Part label for each x_inds element (stable-unique, aligned).
    distance : np.ndarray (float64)
        Updated cumulative distance for each x_inds element (stable-unique, aligned).
        Within a part it increases by exactly step_m per sample. Between parts it also increases by step_m.
    """
    x = np.asarray(x, dtype=np.float64)
    E = np.asarray(easting, dtype=np.float64)
    N = np.asarray(northing, dtype=np.float64)

    n = len(x)
    if n == 0:
        return (np.empty(0, np.uint16), np.empty(0, np.int32), np.empty(0, np.float64))
    if n == 1:
        xi = np.rint(x[:1]).astype(np.int64)
        if (xi < 0).any() or (xi > 65535).any():
            raise ValueError("Rounded x index out of uint16 range [0, 65535].")
        return (xi.astype(np.uint16), np.array([0], np.int32), np.array([0.0], np.float64))

    # Compute segment lengths and jump mask from geometry
    dE = np.diff(E)
    dN = np.diff(N)
    seglen = np.hypot(dE, dN)
    jump = seglen > jump_threshold  # length n-1 boolean

    # Part id per vertex (0,1,2,...)
    part_vertex = np.zeros(n, dtype=np.int32)
    part_vertex[1:] = np.cumsum(jump).astype(np.int32)

    r2 = step_m * step_m

    # We'll collect sampled x (float), part_id (int), distance (float)
    xs = []
    ps = []
    ds = []

    # Helper to append a sample point
    def emit(xpos, part, dist):
        xs.append(xpos)
        ps.append(part)
        ds.append(dist)

    # Start: emit first vertex
    current_part = part_vertex[0]
    dist = 0.0
    Px, Py, Pxpos = E[0], N[0], x[0]
    emit(Pxpos, current_part, dist)

    # Scan along segments, but split on jumps
    i = 0
    # segment start S is always at the current scanning position (initially at the last emitted point)
    Sx, Sy, xS = Px, Py, Pxpos

    while i < n - 1:
        # If this segment is a jump, start new part at vertex i+1
        if jump[i]:
            # Reset to start of next part
            i += 1
            if i >= n:
                break
            current_part = part_vertex[i]
            Px, Py, Pxpos = E[i], N[i], x[i]
            dist += step_m  # cap jump increment to step size (matches your previous approach)
            emit(Pxpos, current_part, dist)

            # restart scanning from this vertex
            Sx, Sy, xS = Px, Py, Pxpos
            continue

        # Normal segment from vertex i to i+1, but start may be inside it (S)
        # Define segment end at vertex i+1
        Tx, Ty, xT = E[i+1], N[i+1], x[i+1]

        dx = Tx - Sx
        dy = Ty - Sy
        a = dx*dx + dy*dy
        if a <= eps:
            # Degenerate sub-segment: advance to next vertex
            i += 1
            if i >= n:
                break
            Sx, Sy, xS = E[i], N[i], x[i]
            continue

        # Circle center is current emitted point P = (Px, Py)
        fx = Sx - Px
        fy = Sy - Py
        b = 2.0 * (dx*fx + dy*fy)
        c = fx*fx + fy*fy - r2
        disc = b*b - 4.0*a*c

        if disc < 0.0:
            # No intersection in this segment; advance to next segment
            i += 1
            if i >= n:
                break
            Sx, Sy, xS = E[i], N[i], x[i]
            continue

        sqrt_disc = np.sqrt(disc)
        inv2a = 1.0 / (2.0*a)
        u1 = (-b - sqrt_disc) * inv2a
        u2 = (-b + sqrt_disc) * inv2a

        # Forward intersections within [0,1], excluding u ~ 0 to avoid repeating the start point
        candidates = []
        if 0.0 <= u1 <= 1.0 and u1 > eps:
            candidates.append(u1)
        if 0.0 <= u2 <= 1.0 and u2 > eps:
            candidates.append(u2)

        if not candidates:
            # Intersection is not forward inside this segment; advance to next segment
            i += 1
            if i >= n:
                break
            Sx, Sy, xS = E[i], N[i], x[i]
            continue

        u = min(candidates)

        # Emit intersection point Q
        Qx = Sx + u * dx
        Qy = Sy + u * dy
        Qxpos = xS + u * (xT - xS)

        dist += step_m
        emit(Qxpos, current_part, dist)

        # Update circle center to Q, and continue scanning from Q on the same segment
        Px, Py, Pxpos = Qx, Qy, Qxpos
        Sx, Sy, xS = Qx, Qy, Qxpos

        # If we're numerically at the segment end, advance i and reset S to the next vertex
        if (Tx - Sx)**2 + (Ty - Sy)**2 <= eps:
            i += 1
            if i >= n:
                break
            Sx, Sy, xS = E[i], N[i], x[i]

    # Convert collected lists to arrays
    xs = np.asarray(xs, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.int32)
    ds = np.asarray(ds, dtype=np.float64)

    # Round x to uint16 indices (with a safety check)
    xi = np.rint(xs).astype(np.int64)
    if (xi < 0).any() or (xi > 65535).any():
        raise ValueError("Rounded x index out of uint16 range [0, 65535]. Consider clipping or revisiting x scale.")
    xi = xi.astype(np.uint16)

    # Stable-unique by x_inds (keep first occurrence), preserving traversal order and alignment.
    # Using a boolean "seen" array is very fast since uint16 range is fixed.
    seen = np.zeros(65536, dtype=bool)
    keep_mask = np.zeros(len(xi), dtype=bool)
    for j, v in enumerate(xi):
        if not seen[v]:
            seen[v] = True
            keep_mask[j] = True

    return xi[keep_mask], ps[keep_mask], ds[keep_mask]

def read_interpretations(radar_key: str, step_m: float) -> pd.DataFrame:
    interp_paths = paths.get_latest_submissions(radar_key)

    with xr.open_dataset(paths.processed_radar_path(radar_key)) as dataset, warnings.catch_warnings():
        depth_model = scipy.interpolate.interp1d(
            np.arange(dataset["data"].shape[0])[::-1],
            dataset["depth"].values,
            bounds_error=False,
        )
        twtt_model = scipy.interpolate.interp1d(
            np.arange(dataset["data"].shape[0])[::-1],
            dataset["return-time"].values,
            bounds_error=False,
        )

        models = {
            key: scipy.interpolate.interp1d(
                dataset["x"].values,
                dataset[key].values,
                bounds_error=False,
            )
            for key in ["easting", "northing", "elevation"]
        }
        x_inds, part_idx, distance = chord_sample_xinds_part_distance(
            x=dataset["x"].values,                # pixel x per vertex
            easting=dataset["easting"].values,
            northing=dataset["northing"].values,
            step_m=step_m,
            jump_threshold=100.0
        )

        models["distance"] = scipy.interpolate.interp1d(x_inds, distance, bounds_error=False)
        models["part_idx"] = scipy.interpolate.interp1d(
            x_inds,
            part_idx,
            bounds_error=False,
            kind="nearest",
        )
        antenna = dataset.attrs["antenna"].split("MHz")[0] + "MHz"
        crs = dataset.attrs["crs"]

    data = pd.concat([read_interpretation_xy(fp, x_vals=x_inds) for fp in interp_paths]).rename(columns={"i": "line_i"})

    data["depth"] = depth_model(data["y"].astype(float))
    data["twtt"] = twtt_model(data["y"].astype(float))
    for key in models:
        data[key] = models[key](data.index.get_level_values("x").astype(float))

    data["distance"] = (data["distance"] / step_m).round() * step_m

    data = (
        data.sort_values("distance")
        .reset_index(level="x", drop=False)
        .set_index("distance", append=True)
        .dropna(subset=["depth", "easting"])
    )
    data = data[~data.index.duplicated(keep="first")]

    data = gpd.GeoDataFrame(data.reset_index(), geometry=gpd.points_from_xy(data["easting"].values, data["northing"].values, crs=crs))

    if crs != "EPSG:32633":
        data = data.to_crs(32633)
        data["easting"] = data.geometry.x
        data["northing"] = data.geometry.y
        
    data["antenna"] = antenna
    data["year"] = radar_key.split("-")[2]
    data["glacier"] = radar_key.split("-")[0]
    return data
