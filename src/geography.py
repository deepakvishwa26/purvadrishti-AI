"""
Geography and ATM reference data generation module — V2.

Generates synthetic ATM reference locations across India.
ATMs are clustered around urban centers to reflect realistic density patterns.

V2 changes:
- Added select_random_atm() for geographic diversity in ATM selection.
- Preserved all existing functions.

All ATM coordinates are SYNTHETIC and do NOT represent real ATM locations.
"""

import numpy as np
import pandas as pd
import h3

from math import radians, sin, cos, sqrt, atan2


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in km between two lat/lon points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def lat_lon_to_h3(lat, lon, resolution=8):
    """Convert latitude/longitude to an H3 cell index."""
    return h3.latlng_to_cell(lat, lon, resolution)


def generate_atm_reference(config, rng):
    """
    Generate a synthetic ATM reference dataset.

    ATMs are distributed with urban clustering: a configurable fraction
    is placed near major urban centers, the remainder scattered across states.

    Args:
        config: Parsed YAML configuration dict.
        rng: numpy RandomState for reproducibility.

    Returns:
        pd.DataFrame with columns: atm_id, latitude, longitude, state, city,
                                    h3_cell_res8, h3_cell_res9, source.
    """
    geo_cfg = config["geography"]
    num_atms = geo_cfg["num_atms"]
    urban_fraction = geo_cfg["urban_atm_fraction"]
    urban_centers = geo_cfg["urban_centers"]
    states_cfg = geo_cfg["states"]

    num_urban = int(num_atms * urban_fraction)
    num_rural = num_atms - num_urban

    atms = []

    # --- Urban ATMs: clustered around city centers ---
    center_weights = np.ones(len(urban_centers)) / len(urban_centers)
    urban_assignments = rng.choice(len(urban_centers), size=num_urban, p=center_weights)

    for idx in urban_assignments:
        center = urban_centers[idx]
        # Generate offset in km, convert to degrees approximately
        radius_km = center["radius_km"]
        angle = rng.uniform(0, 2 * np.pi)
        dist_km = rng.exponential(radius_km / 3)  # Most ATMs near center
        dist_km = min(dist_km, radius_km * 1.5)

        dlat = (dist_km * cos(angle)) / 111.32
        dlon = (dist_km * sin(angle)) / (111.32 * cos(radians(center["lat"])))

        lat = center["lat"] + dlat
        lon = center["lon"] + dlon

        # Determine state from urban center location
        state = _find_state_for_coords(lat, lon, states_cfg)
        atms.append({
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "state": state,
            "city": center["name"],
        })

    # --- Rural/non-urban ATMs: scattered across states ---
    state_names = list(states_cfg.keys())
    state_weights = np.array([states_cfg[s]["weight"] for s in state_names])
    state_weights = state_weights / state_weights.sum()
    rural_state_assignments = rng.choice(len(state_names), size=num_rural, p=state_weights)

    for idx in rural_state_assignments:
        sname = state_names[idx]
        scfg = states_cfg[sname]
        lat = rng.uniform(scfg["lat_range"][0], scfg["lat_range"][1])
        lon = rng.uniform(scfg["lon_range"][0], scfg["lon_range"][1])
        atms.append({
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "state": sname.replace("_", " "),
            "city": None,
        })

    df = pd.DataFrame(atms)
    df["atm_id"] = [f"ATM{str(i + 1).zfill(6)}" for i in range(len(df))]
    df["h3_cell_res8"] = df.apply(lambda r: lat_lon_to_h3(r["latitude"], r["longitude"], 8), axis=1)
    df["h3_cell_res9"] = df.apply(lambda r: lat_lon_to_h3(r["latitude"], r["longitude"], 9), axis=1)
    df["source"] = "SYNTHETIC_REFERENCE"

    cols = ["atm_id", "latitude", "longitude", "state", "city",
            "h3_cell_res8", "h3_cell_res9", "source"]
    return df[cols].reset_index(drop=True)


def _find_state_for_coords(lat, lon, states_cfg):
    """Find the most likely state for given coordinates by bounding-box overlap."""
    for sname, scfg in states_cfg.items():
        lr = scfg["lat_range"]
        lnr = scfg["lon_range"]
        if lr[0] <= lat <= lr[1] and lnr[0] <= lon <= lnr[1]:
            return sname.replace("_", " ")
    return "Unknown"


def compute_atm_density(lat, lon, atm_df, radius_km):
    """
    Count ATMs within a given radius of a point.

    Args:
        lat, lon: Query point.
        atm_df: ATM reference DataFrame with latitude/longitude.
        radius_km: Search radius in kilometers.

    Returns:
        int: Number of ATMs within radius.
    """
    dists = atm_df.apply(
        lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
    )
    return int((dists <= radius_km).sum())


def compute_atm_density_h3(h3_cell, atm_df, resolution=8):
    """
    Count ATMs within the given H3 cell and its immediate neighbors.

    Args:
        h3_cell: Target H3 cell identifier.
        atm_df: ATM reference DataFrame.
        resolution: H3 resolution used.

    Returns:
        int: Number of ATMs in cell + neighbors.
    """
    try:
        neighbors = h3.grid_disk(h3_cell, 1)  # cell + 6 neighbors
    except Exception:
        return 0
    col = f"h3_cell_res{resolution}"
    if col not in atm_df.columns:
        return 0
    return int(atm_df[col].isin(neighbors).sum())


def select_nearby_atm(lat, lon, atm_df, rng, max_distance_km=50):
    """
    Select an ATM near the given coordinates, weighted by proximity.

    Args:
        lat, lon: Reference point (e.g., last mule account location proxy).
        atm_df: ATM reference DataFrame.
        rng: numpy RandomState.
        max_distance_km: Maximum search radius.

    Returns:
        pd.Series: Selected ATM row, or random ATM if none within range.
    """
    dists = atm_df.apply(
        lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1
    )
    nearby = atm_df[dists <= max_distance_km]
    if len(nearby) == 0:
        # Fallback: pick closest 10
        nearest_idx = dists.nsmallest(10).index
        nearby = atm_df.loc[nearest_idx]
        dists = dists.loc[nearest_idx]
    else:
        dists = dists[dists <= max_distance_km]

    # Weight inversely by distance (closer = more likely)
    weights = 1.0 / (dists.loc[nearby.index] + 0.1)
    weights = weights / weights.sum()
    chosen_idx = rng.choice(nearby.index, p=weights.values)
    return atm_df.loc[chosen_idx]


def select_random_atm(atm_df, rng):
    """
    Select a random ATM from the entire reference dataset.
    V2: Used for geographic diversity in ATM selection.

    Args:
        atm_df: ATM reference DataFrame.
        rng: numpy RandomState.

    Returns:
        pd.Series: Selected ATM row.
    """
    idx = rng.randint(0, len(atm_df))
    return atm_df.iloc[idx]
