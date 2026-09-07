"""
SIH 2026 Problem Statement: SIH26011
Prototype: 3D Cadastral & Land Administration Digital Twin
Standard: LADM ISO 19152 (3D Spatial Units)
Rendering: Procedurally Realistic Indian Urban Digital Twin (CesiumJS 3D)
Features: Setbacks, Balconies, Chajjas, Sintex Tanks, Rebar, Auto-rickshaws, 5-Mode Shaders,
          Architectural Slabs, Window Louvers & Reinforced Columns.
"""

import os
import uuid
import dataclasses
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from shapely.geometry import Polygon, Point, box
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

import hashlib
import io
import base64
import datetime
import random as pyrandom

try:
    import qrcode
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False

DEFAULT_REF_LON = 77.5946
DEFAULT_REF_LAT = 12.9716
METERS_PER_DEG_LAT = 111320.0

#==========================================================================================================================
# ============================================================
# LIGHTWEIGHT BACKGROUND VIDEO
# ============================================================

import tempfile
import subprocess
import uuid
import dataclasses

@st.cache_data(show_spinner=False)
import os
import hashlib

def get_web_video_base64(video_path: str) -> str:
    if not video_path or not os.path.exists(video_path):
        return ""

    src_size = os.path.getsize(video_path)

    cache_dir = os.path.join(
        tempfile.gettempdir(),
        "sih26011"
    )
    os.makedirs(cache_dir, exist_ok=True)

    web_video = os.path.join(
        cache_dir,
        "background_web.mp4"
    )

    # Re-encode only if cached version doesn't exist
    # or is suspiciously large.
    if (
        not os.path.exists(web_video)
        or os.path.getsize(web_video) > src_size * 0.65
    ):
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    video_path,

                    # Lightweight browser-friendly video
                    "-vf",
                    "scale='min(1280,iw)':-2,fps=20",

                    # Background doesn't need audio
                    "-an",

                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "30",

                    # Important for browser compatibility
                    "-pix_fmt",
                    "yuv420p",

                    "-movflags",
                    "+faststart",

                    web_video,
                ],
                check=True,
            )

        except (FileNotFoundError, subprocess.CalledProcessError):
            # If ffmpeg isn't installed, use original video.
            web_video = video_path

    try:
        with open(web_video, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return ""


def set_local_bg_video(video_path: str, overlay_opacity: float = 0.55):

    if not video_path or not os.path.exists(video_path):
        return

    b64_video = get_web_video_base64(video_path)

    if not b64_video:
        return

    # NOTE: this used to be injected via st.markdown(..., unsafe_allow_html=True).
    # That renders through React's dangerouslySetInnerHTML, so the <script> block
    # was always inert (script tags inserted that way never execute per the DOM
    # spec), and the <style> block can get scoped away from the real page depending
    # on the Streamlit version. Fix: run this inside components.html's real iframe
    # (where scripts DO execute) and reach into window.parent.document to attach
    # the style/video/overlay directly onto the actual top-level page.
    injector_html = f"""
    <script>
    (function() {{
        const doc = window.parent.document;

        // Streamlit reruns the whole script on every interaction — guard
        // against re-appending duplicate nodes each rerun.
        if (doc.getElementById("sih-bg-video")) return;

        const style = doc.createElement("style");
        style.id = "sih-bg-video-style";
        style.textContent = `
            .stApp {{ background: transparent !important; }}
            [data-testid="stAppViewContainer"] {{ background: transparent !important; }}
            [data-testid="stHeader"] {{ background: transparent !important; }}
            header {{ background: transparent !important; }}
            footer {{ background: transparent !important; }}

            #sih-bg-video {{
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                object-fit: cover;
                pointer-events: none;
                z-index: -2 !important;
                background: #050812;
            }}

            #sih-video-overlay {{
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                background: linear-gradient(
                    90deg,
                    rgba(3, 7, 15, {overlay_opacity + 0.33 if overlay_opacity + 0.33 < 1 else 0.95}),
                    rgba(3, 7, 15, {overlay_opacity}),
                    rgba(3, 7, 15, {overlay_opacity + 0.17 if overlay_opacity + 0.17 < 1 else 0.95})
                );
                pointer-events: none;
                z-index: -1 !important;
            }}

            [data-testid="stSidebar"] {{
                background: rgba(8, 12, 20, 0.82) !important;
                border-right: 1px solid rgba(255,255,255,0.10);
            }}

            h1, h2, h3, h4, h5, h6 {{ color: #F0F6FC !important; }}
            p, label {{ color: #C9D1D9 !important; }}
            [data-testid="stMetricValue"] {{ color: #58FFAA !important; font-weight: 700 !important; }}
            [data-testid="stMetricLabel"] {{ color: #8B949E !important; }}
        `;
        doc.head.appendChild(style);

        const video = doc.createElement("video");
        video.id = "sih-bg-video";
        video.autoplay = true;
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = "auto";
        video.src = "data:video/mp4;base64,{b64_video}";
        doc.body.appendChild(video);

        const overlay = doc.createElement("div");
        overlay.id = "sih-video-overlay";
        doc.body.appendChild(overlay);

        const startVideo = () => {{
            const p = video.play();
            if (p !== undefined) {{
                p.catch(() => {{}});
            }}
        }};

        startVideo();

        doc.addEventListener("visibilitychange", () => {{
            if (!doc.hidden) {{
                startVideo();
            }}
        }});
    }})();
    </script>
    """

    components.html(injector_html, height=0, width=0)


# ============================================================
# ACTIVATE BACKGROUND
# ============================================================

VIDEO_PATH = os.path.join(
    os.path.dirname(__file__),
    "assets",
    "background.mp4"
)

set_local_bg_video(
    VIDEO_PATH,
    overlay_opacity=0.55
)

#===============================================================================================================================
def local_to_wgs84(x: float, y: float, ref_lon: float = DEFAULT_REF_LON, ref_lat: float = DEFAULT_REF_LAT) -> Tuple[float, float]:
    meters_per_lon = 111320.0 * np.cos(np.radians(ref_lat))
    lon = ref_lon + (x / meters_per_lon)
    lat = ref_lat + (y / METERS_PER_DEG_LAT)
    return float(lon), float(lat)


# ============================================================================
# 1. CORE DATA MODEL (LADM ISO 19152)
# ============================================================================

@dataclasses.dataclass
class PropertyUnit:
    unit_id: str
    building_id: str
    floor_id: str
    unit_number: str
    ulpin: str
    area_sqm: float
    volume_cum: float
    property_type: str
    owner_name: str
    rights_type: str
    cx: float
    cy: float
    w: float
    d: float
    z_min: float
    z_max: float


@dataclasses.dataclass
class Floor:
    floor_id: str
    building_id: str
    floor_number: int
    elevation: float
    abs_elevation: float
    height: float
    area_sqm: float
    units: List[PropertyUnit]


@dataclasses.dataclass
class Building:
    building_id: str
    cx: float
    cy: float
    width: float
    depth: float
    height: float
    ground_elevation: float
    num_floors: int
    reconstruction_method: str
    floors: List[Floor]
    typology: str
    color_palette: dict
    roof_style: str
    has_awning: bool
    water_tank_color: str
    has_balconies: bool
    is_under_construction: bool
    num_underground_floors: int = 0


@dataclasses.dataclass
class CityModel:
    city_id: str
    name: str
    buildings: Dict[str, Building]
    ground_datum: float
    ref_lon: float
    ref_lat: float
    road_x_coords: List[float]
    road_y_coords: List[float]
    ground_elements: List[dict]
    vegetation_elements: List[dict]


INDIAN_PALETTES = {
    "weathered_plaster": {"primary": "#A8A29E", "accent": "#00FFAA", "roof": "#44403C"},
    "exposed_brick": {"primary": "#9A3412", "accent": "#FB923C", "roof": "#431407"},
    "painted_terracotta": {"primary": "#C2410C", "accent": "#FDBA74", "roof": "#7C2D12"},
    "monsoon_concrete": {"primary": "#64748B", "accent": "#38BDF8", "roof": "#1E293B"},
    "commercial_glass": {"primary": "#0284C7", "accent": "#FACC15", "roof": "#0F172A"},
    "sandstone_yellow": {"primary": "#D97706", "accent": "#FDE047", "roof": "#78350F"},
    "under_construction": {"primary": "#71717A", "accent": "#E11D48", "roof": "#52525B"}
}


# ============================================================================
# 2. CADASTRAL SUBDIVISION (ISO 19152)
# ============================================================================

def generate_architectural_floorplan(bx: float, by: float, w: float, d: float,
                                     b_id: str, f_id: str, floor_num: int,
                                     z_min: float, floor_h: float, names: list) -> List[PropertyUnit]:
    """
    Builds a floor's cadastral units as a front-wing / central-corridor /
    back-wing layout instead of 4 identical quadrant boxes, so the floor
    plan reads like an actual building (street-facing bays, a shared
    circulation corridor, a stairwell/lift core, and back-of-house or
    rear units) and scales with the footprint instead of always being a
    fixed 2x2 grid.
    """
    units: List[PropertyUnit] = []
    ulpin_prefix = f"ULPIN-KA-{b_id[-4:]}-F{floor_num:02d}"

    def make_unit(counter: int, p_type: str, r_type: str, owner: str,
                  ucx: float, ucy: float, uw: float, ud: float) -> PropertyUnit:
        unit_num = f"{floor_num}{counter:02d}"
        return PropertyUnit(
            unit_id=f"{b_id}-U{unit_num}",
            building_id=b_id, floor_id=f_id, unit_number=unit_num,
            ulpin=f"{ulpin_prefix}-U{counter:02d}",
            area_sqm=round(uw * ud, 2),
            volume_cum=round(uw * ud * max(0.1, floor_h - 0.3), 2),
            property_type=p_type, owner_name=owner, rights_type=r_type,
            cx=ucx, cy=ucy, w=uw, d=ud, z_min=z_min, z_max=z_min + floor_h,
        )

    # Central circulation corridor splits the footprint into a street-facing
    # front wing and a rear wing, mirroring how a real shallow urban plot
    # is organized (front bays + a spine corridor + back-of-house/rear bays).
    corridor_d = max(1.4, min(2.4, d * 0.20))
    wing_d = max(1.5, (d - corridor_d) / 2.0)
    front_cy = by - (corridor_d / 2.0 + wing_d / 2.0)
    back_cy = by + (corridor_d / 2.0 + wing_d / 2.0)

    # Number of bays scales with building width instead of always being 2,
    # so a wide building reads as a real multi-bay facade and a narrow one
    # doesn't get over-subdivided into slivers.
    n_bays = max(2, min(6, round(w / 4.5)))
    bay_step = w / n_bays

    def bay_cx(i: int) -> float:
        return bx - w / 2.0 + bay_step * (i + 0.5)

    counter = 1

    if floor_num == 1:
        # Ground floor: a row of street-facing shops out front...
        for i in range(n_bays):
            label = chr(ord('A') + i) if i < 26 else str(i + 1)
            owner = names[(floor_num * 7 + counter) % len(names)]
            units.append(make_unit(
                counter, f"Ground Commercial Shop {label}", "Commercial Lease", owner,
                bay_cx(i), front_cy, bay_step * 0.92, wing_d * 0.94,
            ))
            counter += 1
        # ...and fewer, larger back-of-house / storage bays at the rear.
        n_rear = max(1, n_bays // 2)
        rear_step = w / n_rear
        for i in range(n_rear):
            units.append(make_unit(
                counter, "Rear Service / Storage", "Freehold Title", "Building Management",
                bx - w / 2.0 + rear_step * (i + 0.5), back_cy, rear_step * 0.92, wing_d * 0.94,
            ))
            counter += 1
    else:
        # Upper floors: flats facing front and back, cycling through
        # 1BHK / 2BHK / 3BHK types so units vary in size like a real
        # residential floor plan rather than four identical squares.
        flat_types = [("Residential 2BHK", 1.00, 1.00), ("Residential 3BHK", 1.18, 1.08),
                      ("Residential 1BHK", 0.78, 0.90)]
        for wing_cy in (front_cy, back_cy):
            for i in range(n_bays):
                p_type, w_mult, d_mult = flat_types[(floor_num + i) % len(flat_types)]
                owner = names[(floor_num * 5 + counter * 3) % len(names)]
                uw = min(bay_step * 0.94, bay_step * 0.94 * w_mult)
                ud = wing_d * 0.94 * min(1.0, d_mult)
                units.append(make_unit(counter, p_type, "Freehold Title", owner, bay_cx(i), wing_cy, uw, ud))
                counter += 1

    # Shared circulation corridor running the full width of the floor.
    corridor_label = "Common Entry Corridor" if floor_num == 1 else "Common Lobby / Corridor"
    units.append(make_unit(
        counter, corridor_label, "Condominium Common", "Apartment Owners Association",
        bx, by, w * 0.92, corridor_d * 0.9,
    ))
    counter += 1

    # Stairwell / lift core: a small dedicated square at one end of the
    # corridor, distinct from the open circulation space around it.
    core_size = max(1.2, min(2.4, corridor_d * 1.1, w * 0.18))
    units.append(make_unit(
        counter, "Common Stairwell / Lift Core", "Condominium Common", "Apartment Owners Association",
        bx - w / 2.0 + core_size / 2.0 + 0.3, by, core_size, core_size,
    ))

    return units


# ============================================================================
# 3. EXTENDED DENSE INDIAN BENCHMARK CITY (64 REALISTIC BUILDINGS)
# ============================================================================

@st.cache_data(show_spinner=False)
def generate_synthetic_city() -> CityModel:
    buildings: Dict[str, Building] = {}
    names = [
        "Kavita Verma", "Ramesh Shah", "Aarav Patel", "Suresh Raina", "Aditi Rao",
        "Sunita Rao", "Rahul Mehta", "Neha Gupta", "Vikram Singh", "Pooja Sharma",
        "Aman Kumar", "Priya Nair", "Deepak Chawla", "Ananya Deshmukh", "Tanvi Kulkarni",
        "Manish Tewari", "Harish Iyer", "Shreya Sen", "Mohammed Rizwan", "Gurpreet Singh"
    ]

    configs = [
        # Sector 1: West Corridor (Dense Mixed Commercial & Residential)
        ("BLD-0001", "weathered_plaster", 6, 6, 14, 16, 3, "water_tank", True, "#0F172A", True, False),
        ("BLD-0002", "exposed_brick", 22, 6, 12, 14, 2, "solar_farm", True, "#F8FAFC", True, False),
        ("BLD-0003", "commercial_glass", 6, 24, 16, 12, 4, "hvac_cube", False, "#0F172A", False, False),
        ("BLD-0004", "painted_terracotta", 24, 24, 11, 13, 3, "water_tank", True, "#0F172A", True, False),
        ("BLD-0005", "under_construction", 7, 40, 8, 8, 2, "water_tank", False, "#0F172A", False, True),
        ("BLD-0006", "sandstone_yellow", 16, 40, 7, 8, 1, "solar_farm", True, "#F8FAFC", False, False),
        ("BLD-0007", "exposed_brick", 24, 40, 8, 8, 2, "water_tank", True, "#0F172A", True, False),

        # Sector 2: Mid-West Urban Blocks
        ("BLD-0008", "monsoon_concrete", 44, 6, 15, 15, 6, "hvac_cube", False, "#0F172A", True, False),
        ("BLD-0009", "painted_terracotta", 61, 6, 13, 13, 4, "water_tank", True, "#0F172A", True, False),
        ("BLD-0010", "weathered_plaster", 44, 23, 12, 15, 3, "water_tank", True, "#F8FAFC", True, False),
        ("BLD-0011", "commercial_glass", 58, 23, 16, 13, 5, "solar_farm", False, "#0F172A", False, False),
        ("BLD-0012", "sandstone_yellow", 44, 40, 7, 8, 1, "water_tank", True, "#0F172A", False, False),
        ("BLD-0013", "monsoon_concrete", 52, 40, 8, 8, 1, "water_tank", True, "#0F172A", False, False),
        ("BLD-0014", "exposed_brick", 61, 40, 7, 8, 2, "water_tank", True, "#F8FAFC", True, False),

        # Sector 3: Central High-Density Hub
        ("BLD-0015", "commercial_glass", 83, 6, 15, 17, 7, "hvac_cube", False, "#0F172A", False, False),
        ("BLD-0016", "sandstone_yellow", 100, 6, 13, 14, 3, "solar_farm", True, "#F8FAFC", True, False),
        ("BLD-0017", "monsoon_concrete", 83, 25, 12, 13, 5, "water_tank", True, "#0F172A", True, False),
        ("BLD-0018", "weathered_plaster", 97, 24, 16, 14, 4, "water_tank", True, "#0F172A", True, False),
        ("BLD-0019", "under_construction", 83, 40, 8, 8, 2, "water_tank", False, "#0F172A", False, True),
        ("BLD-0020", "painted_terracotta", 93, 40, 9, 8, 1, "solar_farm", True, "#F8FAFC", False, False),

        # Sector 4: Residential Strata Blocks
        ("BLD-0021", "monsoon_concrete", 6, 60, 15, 16, 5, "hvac_cube", False, "#0F172A", True, False),
        ("BLD-0022", "weathered_plaster", 23, 60, 13, 14, 3, "water_tank", True, "#0F172A", True, False),
        ("BLD-0023", "commercial_glass", 6, 78, 17, 13, 4, "solar_farm", False, "#0F172A", False, False),
        ("BLD-0024", "sandstone_yellow", 25, 77, 11, 15, 2, "water_tank", True, "#F8FAFC", True, False),
        ("BLD-0025", "exposed_brick", 7, 95, 8, 8, 1, "water_tank", True, "#0F172A", False, False),
        ("BLD-0026", "painted_terracotta", 16, 95, 8, 8, 1, "water_tank", True, "#0F172A", False, False),
        ("BLD-0027", "sandstone_yellow", 25, 95, 8, 8, 2, "solar_farm", True, "#F8FAFC", True, False),

        # Sector 5: Institutional & Mid-rise Flats
        ("BLD-0028", "monsoon_concrete", 44, 60, 15, 17, 8, "hvac_cube", False, "#0F172A", True, False),
        ("BLD-0029", "painted_terracotta", 61, 61, 13, 14, 5, "water_tank", True, "#0F172A", True, False),
        ("BLD-0030", "weathered_plaster", 44, 79, 16, 12, 3, "water_tank", True, "#F8FAFC", True, False),
        ("BLD-0031", "commercial_glass", 62, 78, 12, 15, 4, "solar_farm", False, "#0F172A", False, False),

        # Sector 6: Northeast IT & Commercial Zone
        ("BLD-0032", "sandstone_yellow", 83, 60, 16, 16, 6, "hvac_cube", False, "#0F172A", True, False),
        ("BLD-0033", "weathered_plaster", 101, 61, 13, 14, 4, "water_tank", True, "#0F172A", True, False),
        ("BLD-0034", "exposed_brick", 83, 79, 12, 13, 3, "solar_farm", True, "#F8FAFC", True, False),
        ("BLD-0035", "commercial_glass", 97, 78, 16, 15, 5, "hvac_cube", False, "#0F172A", False, False),
        ("BLD-0036", "painted_terracotta", 83, 95, 8, 8, 1, "water_tank", True, "#0F172A", False, False),
        ("BLD-0037", "sandstone_yellow", 92, 95, 8, 8, 1, "solar_farm", True, "#F8FAFC", False, False),
        ("BLD-0038", "under_construction", 101, 95, 8, 8, 2, "water_tank", False, "#0F172A", False, True),

        # Sector 7: Eastern Tech Extension (EXPANDED DENSE BENCHMARK)
        ("BLD-0039", "commercial_glass", 123, 6, 16, 18, 9, "hvac_cube", False, "#0F172A", False, False),
        ("BLD-0040", "monsoon_concrete", 141, 6, 14, 15, 6, "solar_farm", True, "#F8FAFC", True, False),
        ("BLD-0041", "weathered_plaster", 123, 26, 13, 14, 4, "water_tank", True, "#0F172A", True, False),
        ("BLD-0042", "exposed_brick", 138, 25, 16, 15, 3, "water_tank", True, "#F8FAFC", True, False),
        ("BLD-0043", "sandstone_yellow", 124, 42, 8, 8, 2, "water_tank", True, "#0F172A", False, False),
        ("BLD-0044", "painted_terracotta", 134, 42, 9, 8, 2, "solar_farm", True, "#0F172A", True, False),
        ("BLD-0045", "under_construction", 145, 42, 9, 8, 3, "water_tank", False, "#0F172A", False, True),

        # Sector 8: Southeast Transit Corridors
        ("BLD-0046", "monsoon_concrete", 123, 60, 15, 17, 7, "hvac_cube", False, "#0F172A", True, False),
        ("BLD-0047", "commercial_glass", 140, 60, 16, 16, 8, "solar_farm", False, "#0F172A", False, False),
        ("BLD-0048", "weathered_plaster", 123, 79, 14, 14, 4, "water_tank", True, "#0F172A", True, False),
        ("BLD-0049", "sandstone_yellow", 139, 79, 15, 15, 3, "water_tank", True, "#F8FAFC", True, False),
        ("BLD-0050", "painted_terracotta", 124, 96, 9, 8, 2, "water_tank", True, "#0F172A", True, False),
        ("BLD-0051", "exposed_brick", 135, 96, 8, 8, 1, "solar_farm", True, "#F8FAFC", False, False),
        ("BLD-0052", "under_construction", 145, 96, 8, 8, 2, "water_tank", False, "#0F172A", False, True),

        # Sector 9: Northern Civic Arterial & Multi-family Stratum
        ("BLD-0053", "sandstone_yellow", 6, 114, 14, 15, 3, "water_tank", True, "#0F172A", True, False),
        ("BLD-0054", "exposed_brick", 22, 114, 13, 14, 3, "solar_farm", True, "#F8FAFC", True, False),
        ("BLD-0055", "monsoon_concrete", 44, 114, 16, 16, 6, "hvac_cube", False, "#0F172A", True, False),
        ("BLD-0056", "commercial_glass", 62, 114, 14, 15, 5, "solar_farm", False, "#0F172A", False, False),
        ("BLD-0057", "weathered_plaster", 83, 114, 15, 16, 4, "water_tank", True, "#0F172A", True, False),
        ("BLD-0058", "painted_terracotta", 100, 114, 14, 14, 3, "solar_farm", True, "#F8FAFC", True, False),
        ("BLD-0059", "monsoon_concrete", 123, 114, 17, 16, 6, "hvac_cube", False, "#0F172A", True, False),
        ("BLD-0060", "commercial_glass", 142, 114, 14, 15, 7, "solar_farm", False, "#0F172A", False, False),

        # Sector 10: Infill Micro-Parcels & Slender Townhomes
        ("BLD-0061", "exposed_brick", 34, 95, 7, 7, 2, "water_tank", True, "#0F172A", False, False),
        ("BLD-0062", "sandstone_yellow", 72, 95, 7, 7, 2, "water_tank", True, "#F8FAFC", False, False),
        ("BLD-0063", "weathered_plaster", 112, 95, 7, 7, 1, "solar_farm", True, "#0F172A", False, False),
        ("BLD-0064", "monsoon_concrete", 112, 40, 8, 8, 2, "water_tank", True, "#0F172A", True, False),
    ]

    for (b_id, pal_key, bx, by, w, d, floors_cnt, r_style, awning, tank_col, balc, under_c) in configs:
        cx = bx + w / 2.0
        cy = by + d / 2.0
        floor_h = 3.2
        b_height = floors_cnt * floor_h
        floor_objs: List[Floor] = []

        for f in range(1, floors_cnt + 1):
            f_id = f"{b_id}-FLR{f:02d}"
            f_elev = (f - 1) * floor_h
            units = generate_architectural_floorplan(cx, cy, w, d, b_id, f_id, f, f_elev, floor_h, names)
            floor_objs.append(Floor(
                floor_id=f_id,
                building_id=b_id,
                floor_number=f,
                elevation=round(f_elev, 2),
                abs_elevation=round(f_elev, 2),
                height=floor_h,
                area_sqm=round(w * d, 2),
                units=units
            ))

        buildings[b_id] = Building(
            building_id=b_id,
            cx=cx,
            cy=cy,
            width=w,
            depth=d,
            height=round(b_height, 2),
            ground_elevation=0.0,
            num_floors=floors_cnt,
            reconstruction_method="Survey Cadastre Boundary (ISO 19152)",
            floors=floor_objs,
            typology=pal_key,
            color_palette=INDIAN_PALETTES[pal_key],
            roof_style=r_style,
            has_awning=awning,
            water_tank_color=tank_col,
            has_balconies=balc,
            is_under_construction=under_c,
            # Deterministic demo variety: roughly 1 in 3 buildings gets a
            # basement/underground level so "Basement / Underground Floors"
            # has something real to show out of the box. Editable per
            # building afterwards from the Structure Editor workspace.
            num_underground_floors=(1 if (int(b_id.split('-')[-1]) % 3 == 0 and pal_key != "under_construction") else 0)
        )

    # Expanded vegetation clusters along setbacks & sidewalks
    synth_veg = []
    tree_coords = [
        (35.0, 15.0, 5.2, "#15803D"), (35.0, 48.0, 1.4, "#166534"),
        (74.0, 12.0, 5.8, "#15803D"), (74.0, 68.0, 1.8, "#166534"),
        (2.0, 52.0, 4.5, "#15803D"), (40.0, 105.0, 4.8, "#166534"),
        (115.0, 15.0, 5.5, "#15803D"), (115.0, 52.0, 1.6, "#166534"),
        (115.0, 88.0, 5.0, "#15803D"), (155.0, 35.0, 4.8, "#166534"),
        (75.0, 125.0, 5.2, "#15803D"), (125.0, 125.0, 5.4, "#166534")
    ]
    for tx, ty, th, tc in tree_coords:
        t_lon, t_lat = local_to_wgs84(tx, ty, DEFAULT_REF_LON, DEFAULT_REF_LAT)
        synth_veg.append({
            "lon": t_lon, "lat": t_lat, "z": 0.0,
            "h": th, "radius": min(2.5, th * 0.45), "color": tc
        })

    g_lon, g_lat = local_to_wgs84(80.0, 70.0, DEFAULT_REF_LON, DEFAULT_REF_LAT)
    synth_ground = [{
        "lon": g_lon, "lat": g_lat, "w": 180.0, "d": 160.0, "z": -0.15, "h": 0.15, "color": "#1E293B"
    }]

    return CityModel(
        city_id="CITY-BLR-SMART-EXT",
        name="Bengaluru Mega Cadastre Twin (ISO 19152)",
        buildings=buildings,
        ground_datum=0.0,
        ref_lon=DEFAULT_REF_LON,
        ref_lat=DEFAULT_REF_LAT,
        road_x_coords=[38.0, 77.0, 117.0],
        road_y_coords=[38.0, 57.0, 107.0],
        ground_elements=synth_ground,
        vegetation_elements=synth_veg
    )

# ============================================================================
# 4. ROBUST MULTI-CLASS LIDAR PIPELINE (LAS / LAZ)
# ============================================================================

def process_uploaded_las(file_bytes: bytes, filename: str) -> CityModel:
    if not HAS_LASPY:
        raise RuntimeError("laspy is missing. Install via: pip install laspy[lazrs]")

    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", os.path.basename(filename))
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    las = laspy.read(temp_path)
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])

    if len(xyz) == 0:
        raise RuntimeError("Point cloud has 0 points.")

    min_x, max_x = float(np.min(xyz[:, 0])), float(np.max(xyz[:, 0]))
    min_y, max_y = float(np.min(xyz[:, 1])), float(np.max(xyz[:, 1]))
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    if -180.0 <= min_x <= 180.0 and -90.0 <= min_y <= 90.0:
        ref_lon = float(center_x)
        ref_lat = float(center_y)
        cos_lat = np.cos(np.radians(ref_lat))
        xyz[:, 0] = (xyz[:, 0] - ref_lon) * 111320.0 * cos_lat
        xyz[:, 1] = (xyz[:, 1] - ref_lat) * 111320.0
    else:
        ref_lon = DEFAULT_REF_LON
        ref_lat = DEFAULT_REF_LAT
        xyz[:, 0] -= center_x
        xyz[:, 1] -= center_y

    classification = np.asarray(las.classification) if hasattr(las, "classification") else np.zeros(len(xyz))
    ground_mask = classification == 2
    low_veg_mask = classification == 3
    high_veg_mask = classification == 5
    building_mask = classification == 6

    ground_pts = xyz[ground_mask]
    z_datum = float(np.percentile(ground_pts[:, 2], 10)) if len(ground_pts) > 0 else float(np.percentile(xyz[:, 2], 5))
    xyz[:, 2] -= z_datum

    building_pts = xyz[building_mask]
    low_veg_pts = xyz[low_veg_mask]
    high_veg_pts = xyz[high_veg_mask]

    ground_elements = []
    if len(ground_pts) > 0:
        step = max(1, len(ground_pts) // 120)
        for pt in ground_pts[::step]:
            gx, gy, _ = pt
            glon, glat = local_to_wgs84(gx, gy, ref_lon, ref_lat)
            ground_elements.append({
                "lon": glon, "lat": glat, "w": 4.5, "d": 4.5, "z": -0.1, "h": 0.1, "color": "#1E293B"
            })

    vegetation_elements = []
    if len(low_veg_pts) > 0:
        step = max(1, len(low_veg_pts) // 70)
        for pt in low_veg_pts[::step]:
            vx, vy, vz = pt
            vlon, vlat = local_to_wgs84(vx, vy, ref_lon, ref_lat)
            vh = float(np.clip(vz, 0.7, 1.8))
            vegetation_elements.append({
                "lon": vlon, "lat": vlat, "z": 0.0, "h": vh, "radius": vh * 0.45, "color": "#166534"
            })

    if len(high_veg_pts) > 0:
        step = max(1, len(high_veg_pts) // 70)
        for pt in high_veg_pts[::step]:
            vx, vy, vz = pt
            vlon, vlat = local_to_wgs84(vx, vy, ref_lon, ref_lat)
            vh = float(np.clip(vz, 3.5, 6.5))
            vegetation_elements.append({
                "lon": vlon, "lat": vlat, "z": 0.0, "h": vh, "radius": min(2.5, vh * 0.45), "color": "#15803D"
            })

    if len(building_pts) < 30:
        elevated_mask = xyz[:, 2] > 2.0
        building_pts = xyz[elevated_mask]

    if len(building_pts) < 15:
        raise RuntimeError("No elevated building points detected in point cloud.")

    if len(building_pts) > 25000:
        step = len(building_pts) // 25000
        building_pts = building_pts[::step]

    clustering = DBSCAN(eps=4.5, min_samples=8, n_jobs=-1).fit(building_pts[:, :2])
    labels = clustering.labels_

    buildings: Dict[str, Building] = {}
    names = ["Kavita Verma", "Ramesh Shah", "Aarav Patel", "Suresh Raina", "Aditi Rao", "Sunita Rao"]
    palette_keys = list(INDIAN_PALETTES.keys())
    b_idx = 1

    unique_lbls = [l for l in np.unique(labels) if l != -1]
    if len(unique_lbls) == 0:
        unique_lbls = [0]
        labels = np.zeros(len(building_pts), dtype=int)

    for lbl in unique_lbls:
        c_pts = building_pts[labels == lbl]
        if len(c_pts) < 8:
            continue

        min_bx, max_bx = float(np.min(c_pts[:, 0])), float(np.max(c_pts[:, 0]))
        min_by, max_by = float(np.min(c_pts[:, 1])), float(np.max(c_pts[:, 1]))
        w = max(5.0, max_bx - min_bx)
        d = max(5.0, max_by - min_by)
        cx = (min_bx + max_bx) / 2.0
        cy = (min_by + max_by) / 2.0

        height = max(3.5, float(np.percentile(c_pts[:, 2], 98)))
        floor_h = 3.3
        num_floors = max(1, min(int(round(height / floor_h)), 60))
        actual_flr_h = height / num_floors
        b_id = f"BLD-SURVEY-{b_idx:04d}"
        pal_name = palette_keys[b_idx % len(palette_keys)]

        floor_objs: List[Floor] = []
        for f in range(1, num_floors + 1):
            f_id = f"{b_id}-FLR{f:02d}"
            f_elev = (f - 1) * actual_flr_h
            units = generate_architectural_floorplan(cx, cy, w, d, b_id, f_id, f, f_elev, actual_flr_h, names)
            floor_objs.append(Floor(
                floor_id=f_id,
                building_id=b_id,
                floor_number=f,
                elevation=round(f_elev, 2),
                abs_elevation=round(f_elev, 2),
                height=round(actual_flr_h, 2),
                area_sqm=round(w * d, 2),
                units=units
            ))

        buildings[b_id] = Building(
            building_id=b_id,
            cx=cx,
            cy=cy,
            width=w,
            depth=d,
            height=round(height, 2),
            ground_elevation=0.0,
            num_floors=num_floors,
            reconstruction_method="Classified LiDAR Extraction (Class 6)",
            floors=floor_objs,
            typology=pal_name,
            color_palette=INDIAN_PALETTES[pal_name],
            roof_style="water_tank",
            has_awning=(b_idx % 2 == 0),
            water_tank_color="#0F172A",
            has_balconies=(b_idx % 2 == 1),
            is_under_construction=(b_idx % 5 == 0)
        )
        b_idx += 1

    if len(buildings) == 0:
        raise RuntimeError("No building bounds could be recovered from point clusters.")

    return CityModel(
        city_id=f"CITY-LIDAR-{uuid.uuid4().hex[:6].upper()}",
        name=f"Survey: {filename}",
        buildings=buildings,
        ground_datum=round(z_datum, 2),
        ref_lon=ref_lon,
        ref_lat=ref_lat,
        road_x_coords=[],
        road_y_coords=[],
        ground_elements=ground_elements,
        vegetation_elements=vegetation_elements
    )


# ============================================================================
# 5. STREET INFRASTRUCTURE, URBAN CLUTTER & DETAIL GENERATOR
# ============================================================================

def compile_infrastructure_elements(city: CityModel) -> Tuple[List[dict], List[dict]]:
    ref_lon, ref_lat = city.ref_lon, city.ref_lat
    infra_list: List[dict] = []
    clutter_list: List[dict] = []

    road_w = 7.0
    sw_w = 2.2
    y_span = 150.0
    x_span = 175.0

    # Primary Arterials (X-corridors)
    for xr in city.road_x_coords:
        r_lon, r_lat = local_to_wgs84(xr, 70.0, ref_lon, ref_lat)
        infra_list.append({
            "lon": r_lon, "lat": r_lat, "w": road_w, "d": y_span, "z": 0.0, "h": 0.08,
            "color": "#1E293B"
        })
        for sw_x in [xr - road_w / 2 - sw_w / 2, xr + road_w / 2 + sw_w / 2]:
            s_lon, s_lat = local_to_wgs84(sw_x, 70.0, ref_lon, ref_lat)
            infra_list.append({
                "lon": s_lon, "lat": s_lat, "w": sw_w, "d": y_span, "z": 0.08, "h": 0.16,
                "color": "#475569"
            })
        for yd in np.arange(0.0, y_span, 7.5):
            d_lon, d_lat = local_to_wgs84(xr, yd + 2.0, ref_lon, ref_lat)
            infra_list.append({
                "lon": d_lon, "lat": d_lat, "w": 0.25, "d": 3.8, "z": 0.09, "h": 0.02,
                "color": "#FACC15"
            })
        for yp in np.arange(10.0, y_span - 10.0, 24.0):
            p_lon, p_lat = local_to_wgs84(xr + road_w / 2 + 0.8, yp, ref_lon, ref_lat)
            clutter_list.append({
                "shape": "cylinder",
                "lon": p_lon, "lat": p_lat, "z": 0.24, "h": 5.2, "radius": 0.12,
                "color": "#94A3B8"
            })

    # Cross Streets (Y-corridors)
    for yr in city.road_y_coords:
        r_lon, r_lat = local_to_wgs84(85.0, yr, ref_lon, ref_lat)
        infra_list.append({
            "lon": r_lon, "lat": r_lat, "w": x_span, "d": road_w, "z": 0.0, "h": 0.08,
            "color": "#1E293B"
        })

    # Auto-Rickshaws (Yellow & Green dual tone volumes)
    rickshaw_coords = [
        (35.8, 28.0), (74.8, 34.0), (41.0, 54.5), (79.5, 36.0),
        (114.5, 32.0), (119.5, 65.0), (74.5, 102.0), (35.5, 102.0)
    ]
    for rx, ry in rickshaw_coords:
        a_lon, a_lat = local_to_wgs84(rx, ry, ref_lon, ref_lat)
        clutter_list.append({
            "shape": "box",
            "lon": a_lon, "lat": a_lat, "w": 1.6, "d": 2.6, "z": 0.08, "h": 1.5,
            "color": "#FACC15"
        })

    # Indian Street Handcarts / Thelas
    cart_coords = [(35.5, 42.0), (74.2, 50.0), (114.2, 48.0), (74.5, 110.0)]
    for cx, cy in cart_coords:
        c_lon, c_lat = local_to_wgs84(cx, cy, ref_lon, ref_lat)
        clutter_list.append({
            "shape": "box",
            "lon": c_lon, "lat": c_lat, "w": 1.2, "d": 2.0, "z": 0.08, "h": 0.85,
            "color": "#78350F"
        })

    # Rooftop Assets: Water Tanks, Balconies, Chajjas, Rebars
    for b_id, bld in city.buildings.items():
        cx, cy = bld.cx, bld.cy
        top_z = bld.height

        # Sintex Water Tank
        w_lon, w_lat = local_to_wgs84(cx - bld.width / 4, cy - bld.depth / 4, ref_lon, ref_lat)
        clutter_list.append({
            "shape": "cylinder",
            "lon": w_lon, "lat": w_lat,
            "z": top_z + 0.6, "h": 1.8, "radius": 1.2,
            "color": bld.water_tank_color
        })

        # Stairhead Mumty Room
        p_lon, p_lat = local_to_wgs84(cx + bld.width / 4, cy + bld.depth / 4, ref_lon, ref_lat)
        clutter_list.append({
            "shape": "box",
            "lon": p_lon, "lat": p_lat,
            "w": 3.4, "d": 3.4, "z": top_z + 0.6, "h": 2.2,
            "color": "#475569"
        })

        # Rebar Pillars on Unfinished Roofs
        if bld.is_under_construction:
            for ox, oy in [(-bld.width / 3, -bld.depth / 3), (bld.width / 3, -bld.depth / 3),
                           (-bld.width / 3, bld.depth / 3), (bld.width / 3, bld.depth / 3)]:
                rb_lon, rb_lat = local_to_wgs84(cx + ox, cy + oy, ref_lon, ref_lat)
                clutter_list.append({
                    "shape": "cylinder",
                    "lon": rb_lon, "lat": rb_lat,
                    "z": top_z + 0.6, "h": 1.8, "radius": 0.08,
                    "color": "#991B1B"
                })

        # Commercial Awning
        if bld.has_awning:
            a_lon, a_lat = local_to_wgs84(cx, cy - bld.depth / 2 - 0.7, ref_lon, ref_lat)
            clutter_list.append({
                "shape": "box",
                "lon": a_lon, "lat": a_lat,
                "w": bld.width + 0.6, "d": 1.4, "z": 3.1, "h": 0.25,
                "color": "#D97706"
            })

        # Cantilevered Balconies along Upper Floors
        if bld.has_balconies and bld.num_floors > 1:
            for f in range(2, bld.num_floors + 1):
                fz = (f - 1) * 3.2
                b_lon, b_lat = local_to_wgs84(cx, cy - bld.depth / 2 - 0.5, ref_lon, ref_lat)
                clutter_list.append({
                    "shape": "box",
                    "lon": b_lon, "lat": b_lat,
                    "w": bld.width * 0.7, "d": 1.0, "z": fz, "h": 0.2,
                    "color": "#475569"
                })
                clutter_list.append({
                    "shape": "box",
                    "lon": b_lon, "lat": b_lat,
                    "w": bld.width * 0.7, "d": 0.1, "z": fz + 0.2, "h": 0.85,
                    "color": "#94A3B8"
                })

    return infra_list, clutter_list


# ============================================================================
# 6. CESIUM COMPONENT BRIDGE (DETAILED BUILDING ARCHITECTURE)
# ============================================================================

COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cesium_component")
_cesium_component = components.declare_component("cesium_viewer", path=COMPONENT_DIR)

import os
import glob
import random
from typing import Optional
import numpy as np

def render_cesium_viewer(
    city: CityModel,
    level: str,
    render_mode: str = "REALISTIC",
    selected_bld_id: Optional[str] = None,
    selected_flr_id: Optional[str] = None,
    selected_unit_id: Optional[str] = None,
    exploded: bool = False,
    cutaway: bool = False,
    show_ground: bool = True,
    show_vegetation: bool = True,
    status_map: Optional[Dict[str, str]] = None,
    overlay_boxes: Optional[List[dict]] = None,
    overlay_points: Optional[List[dict]] = None,
    imported_floor_ids: Optional[set] = None,
) -> Optional[dict]:
    # -------------------------------------------------------------------------
    # 0. INLINE SAFE TEXTURE RESOLVER (Zero-dependency fallback)
    # -------------------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    asset_base = os.path.join(base_dir, "assets")

    typology_subfolder_map = {
        "weathered_plaster": "residential",
        "exposed_brick": "brick",
        "painted_terracotta": "residential",
        "monsoon_concrete": "apartment",
        "commercial_glass": "commercial",
        "sandstone_yellow": "residential",
        "under_construction": "unfinished"
    }

    def resolve_facade_slice(typology_key: str, floor_index: int = 1, is_awning_active: bool = False) -> Optional[str]:
        if not os.path.isdir(asset_base):
            return None

        # Ground floors with awnings map to commercial stalls/shops
        if floor_index == 1 and is_awning_active:
            sub = "commercial"
        else:
            sub = typology_subfolder_map.get(typology_key, "residential")

        target_folder = os.path.join(asset_base, "facades", sub)
        if os.path.isdir(target_folder):
            extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
            matched = []
            for ext in extensions:
                matched.extend(glob.glob(os.path.join(target_folder, ext)))

            if matched:
                # Deterministic selection so building facades don't flutter on rerun
                deterministic_seed = f"{typology_key}_{sub}_{floor_index}"
                rng = random.Random(deterministic_seed)
                chosen_abs = rng.choice(matched)
                return os.path.relpath(chosen_abs, base_dir).replace("\\", "/")

        return None

    def resolve_roof_tile(typology_key: str) -> Optional[str]:
        if not os.path.isdir(asset_base):
            return None

        roof_folder = os.path.join(asset_base, "roofs")
        if os.path.isdir(roof_folder):
            extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
            matched = []
            for ext in extensions:
                matched.extend(glob.glob(os.path.join(roof_folder, ext)))

            if matched:
                rng = random.Random(typology_key)
                chosen_abs = rng.choice(matched)
                return os.path.relpath(chosen_abs, base_dir).replace("\\", "/")

        return None

    boxes = []
    # Floors that already have a converted/imported floor plan (image or DXF)
    # standing in for them — their default generated geometry is suppressed
    # below so only the imported walls are shown, not both stacked together.
    _imported_flr_ids = imported_floor_ids or set()
    ref_lon, ref_lat = city.ref_lon, city.ref_lat
    infra_data, clutter_data = compile_infrastructure_elements(city)

    # -------------------------------------------------------------------------
    # 1. CITY LEVEL VIEW
    # -------------------------------------------------------------------------
    if level == "CITY":
        for b_id, bld in city.buildings.items():
            is_sel = (b_id == selected_bld_id)
            w_lon, w_lat = local_to_wgs84(bld.cx, bld.cy, ref_lon, ref_lat)

            # Texture resolution with dynamic UV coordinate ratios
            city_facade_tex = resolve_facade_slice(bld.typology, floor_index=1, is_awning_active=bld.has_awning)
            city_roof_tex = resolve_roof_tile(bld.typology)
            horizontal_repeat = max(1.0, round((bld.width + bld.depth) / 8.0))

            boxes.append({
                "id": b_id,
                "name": f"Building {b_id}",
                "level": "CITY",
                "lon": w_lon,
                "lat": w_lat,
                "w": bld.width,
                "d": bld.depth,
                "base_z": 0.0,
                "h": bld.height,
                "color": "#0284C7" if is_sel else bld.color_palette["primary"],
                "roof_color": bld.color_palette["roof"],
                "texture_url": city_facade_tex,
                "roof_texture_url": city_roof_tex,
                "repeat_x": horizontal_repeat,
                "repeat_y": float(bld.num_floors),
                "is_selected": is_sel,
                "metadata": {
                    "Building ID": b_id,
                    "Typology": bld.typology.replace("_", " ").title(),
                    "Floors": bld.num_floors,
                    "Height": f"{bld.height:.1f} m",
                    "Tenure Class": "Freehold Title" if "plaster" in bld.typology or "brick" in bld.typology else "Commercial Lease",
                    "Status": dominant_building_status(bld, status_map) if status_map else "Verified",
                }
            })

    # -------------------------------------------------------------------------
    # 2. BUILDING ARCHITECTURAL DETAIL LEVEL VIEW
    # -------------------------------------------------------------------------
    elif level == "BUILDING" and selected_bld_id in city.buildings:
        bld = city.buildings[selected_bld_id]
        w_lon, w_lat = local_to_wgs84(bld.cx, bld.cy, ref_lon, ref_lat)

        # A. Foundation Plinth Slab
        boxes.append({
            "id": f"{bld.building_id}-PLINTH",
            "name": f"{bld.building_id} Ground Plinth",
            "level": "DETAIL",
            "lon": w_lon, "lat": w_lat,
            "w": bld.width + 0.8, "d": bld.depth + 0.8,
            "base_z": -0.3, "h": 0.35,
            "color": "#1E293B", "roof_color": "#0F172A",
            "is_selected": False,
            "metadata": {"Component": "Structural Plinth (IS 1904)"}
        })

        for fi, flr in enumerate(bld.floors):
            is_flr_sel = (flr.floor_id == selected_flr_id)
            gap = (fi * 3.8) if exploded else 0.0
            base_z = flr.elevation + gap
            setback_factor = 0.90 if (fi == len(bld.floors) - 1 and len(bld.floors) > 2) else 1.0

            if flr.floor_id in _imported_flr_ids:
                # A converted floor plan (image or DXF) already stands in for
                # this floor — skip its default core/slab/columns/units so it
                # doesn't render underneath/behind the imported walls.
                continue

            if cutaway:
                unit_palette = ["#0284C7", "#10B981", "#E11D48", "#8B5CF6", "#D97706"]
                for ui, unit in enumerate(flr.units):
                    u_lon, u_lat = local_to_wgs84(unit.cx, unit.cy, ref_lon, ref_lat)
                    boxes.append({
                        "id": unit.unit_id,
                        "name": f"Unit {unit.unit_number}",
                        "level": "UNIT",
                        "lon": u_lon, "lat": u_lat,
                        "w": unit.w * setback_factor,
                        "d": unit.d * setback_factor,
                        "base_z": base_z + 0.15,
                        "h": flr.height - 0.3,
                        "color": unit_palette[ui % len(unit_palette)],
                        "roof_color": "#1E293B",
                        "is_selected": False,
                        "metadata": {
                            "Unit ID": unit.unit_id,
                            "Owner": unit.owner_name,
                            "Title": unit.rights_type,
                            "Tenure Class": unit.rights_type,
                            "Area": f"{unit.area_sqm} m²",
                            "Status": (status_map or {}).get(unit.unit_id, "Verified"),
                        }
                    })
            else:
                # Photographic floor slice assignment
                floor_facade_tex = resolve_facade_slice(
                    bld.typology,
                    floor_index=flr.floor_number,
                    is_awning_active=bld.has_awning
                )
                repeat_u = max(1.0, round(bld.width / 6.0))

                # Architectural Recessed Glazing Core / Textured Floor
                boxes.append({
                    "id": flr.floor_id,
                    "name": f"Floor {flr.floor_number} Core",
                    "level": "FLOOR",
                    "lon": w_lon, "lat": w_lat,
                    "w": (bld.width - 0.3) * setback_factor,
                    "d": (bld.depth - 0.3) * setback_factor,
                    "base_z": base_z + 0.22,
                    "h": flr.height - 0.28,
                    "color": "#0284C7" if bld.typology == "commercial_glass" else bld.color_palette["primary"],
                    "roof_color": bld.color_palette["roof"],
                    "texture_url": floor_facade_tex,
                    "repeat_x": repeat_u,
                    "repeat_y": 1.0,
                    "is_selected": is_flr_sel,
                    "metadata": {
                        "Floor ID": flr.floor_id,
                        "Elevation": f"{flr.elevation:.1f} m",
                        "Parcels": len(flr.units),
                        "Tenure Class": "Strata Title"
                    }
                })

                # Projecting RCC Floor Slab Band
                boxes.append({
                    "id": f"{flr.floor_id}-SLAB",
                    "name": f"Floor {flr.floor_number} Slab Band",
                    "level": "DETAIL",
                    "lon": w_lon, "lat": w_lat,
                    "w": (bld.width + 0.25) * setback_factor,
                    "d": (bld.depth + 0.25) * setback_factor,
                    "base_z": base_z,
                    "h": 0.22,
                    "color": "#334155",
                    "roof_color": bld.color_palette["roof"],
                    "is_selected": is_flr_sel,
                    "metadata": {"Component": "Reinforced Slab"}
                })

                # Load-Bearing Corner Pillars
                for cx_mult in [-0.5, 0.5]:
                    for cy_mult in [-0.5, 0.5]:
                        col_x = bld.cx + (bld.width * setback_factor * cx_mult)
                        col_y = bld.cy + (bld.depth * setback_factor * cy_mult)
                        c_lon, c_lat = local_to_wgs84(col_x, col_y, ref_lon, ref_lat)
                        boxes.append({
                            "id": f"{flr.floor_id}-COL-{cx_mult}-{cy_mult}",
                            "name": "Structural Column",
                            "level": "DETAIL",
                            "lon": c_lon, "lat": c_lat,
                            "w": 0.55, "d": 0.55,
                            "base_z": base_z,
                            "h": flr.height,
                            "color": bld.color_palette["accent"],
                            "roof_color": bld.color_palette["roof"],
                            "is_selected": False,
                            "metadata": {"Component": "RCC Column (IS 456)"}
                        })

        # Terrace Parapet Cap
        top_z = bld.height + ((len(bld.floors) - 1) * 3.8 if exploded else 0.0)
        boxes.append({
            "id": f"{bld.building_id}-PARAPET",
            "name": "Terrace Parapet",
            "level": "DETAIL",
            "lon": w_lon, "lat": w_lat,
            "w": bld.width + 0.15, "d": bld.depth + 0.15,
            "base_z": top_z, "h": 0.9,
            "color": bld.color_palette["primary"],
            "roof_color": bld.color_palette["roof"],
            "roof_texture_url": resolve_roof_tile(bld.typology),
            "is_selected": False,
            "metadata": {"Type": "Safety Parapet (1.0m)"}
        })

    # -------------------------------------------------------------------------
    # 3. FLOOR / STRATA UNIT PARCEL LEVEL VIEW
    # -------------------------------------------------------------------------
    elif level in ["FLOOR", "UNIT"] and selected_bld_id in city.buildings:
        bld = city.buildings[selected_bld_id]
        flr = next((f for f in bld.floors if f.floor_id == selected_flr_id), bld.floors[0])

        unit_palette = ["#0284C7", "#10B981", "#E11D48", "#8B5CF6", "#D97706"]
        _skip_default_units = flr.floor_id in _imported_flr_ids
        for ui, unit in enumerate([] if _skip_default_units else flr.units):
            is_u_sel = (unit.unit_id == selected_unit_id)
            u_lon, u_lat = local_to_wgs84(unit.cx, unit.cy, ref_lon, ref_lat)

            boxes.append({
                "id": unit.unit_id,
                "name": f"Unit {unit.unit_number}",
                "level": "UNIT",
                "lon": u_lon,
                "lat": u_lat,
                "w": unit.w,
                "d": unit.d,
                "base_z": 0.0,
                "h": 3.0,
                "color": "#00FFAA" if is_u_sel else unit_palette[ui % len(unit_palette)],
                "roof_color": "#0F172A",
                "is_selected": is_u_sel,
                "metadata": {
                    "Unit ID": unit.unit_id,
                    "ULPIN": unit.ulpin,
                    "Owner": unit.owner_name,
                    "Tenure Class": unit.rights_type,
                    "Area": f"{unit.area_sqm} m²",
                    "Status": (status_map or {}).get(unit.unit_id, "Verified"),
                }
            })

    # Dynamic camera framing
    if boxes:
        target_lon = float(np.mean([b["lon"] for b in boxes]))
        target_lat = float(np.mean([b["lat"] for b in boxes]))
        target_range = 480.0 if level == "CITY" else (70.0 if level == "BUILDING" else 35.0)
    else:
        target_lon, target_lat, target_range = ref_lon, ref_lat, 450.0

    component_data = {
        "boxes": boxes,
        "infrastructure": infra_data if level == "CITY" else [],
        "clutter": clutter_data if level == "CITY" else [],
        "ground": city.ground_elements if level == "CITY" else [],
        "vegetation": city.vegetation_elements if level == "CITY" else [],
        "render_mode": render_mode,
        "show_ground": show_ground,
        "show_vegetation": show_vegetation,
        "target_lon": target_lon,
        "target_lat": target_lat,
        "target_range": target_range,
        # Additive analytical overlay layers (Section 8) — empty by default,
        # so the original pipeline's visuals are completely unaffected.
        "overlay_boxes": overlay_boxes or [],
        "overlay_points": overlay_points or [],
    }

    overlay_sig = f"{len(overlay_boxes or [])}-{len(overlay_points or [])}-{len(_imported_flr_ids)}"
    return _cesium_component(
        data=component_data,
        key=f"cesium-{level}-{render_mode}-{selected_bld_id}-{selected_flr_id}-{selected_unit_id}-{exploded}-{cutaway}-{show_ground}-{show_vegetation}-{overlay_sig}"
    )



# ============================================================================
# 6.5 SIH26011 FIELD / ADMINISTRATION WORKSPACE
# ============================================================================
def _all_units(city: CityModel):
    rows = []
    for b in city.buildings.values():
        for f in b.floors:
            for u in f.units:
                rows.append((b, f, u))
    return rows


def cadastral_statistics(city: CityModel):
    units = _all_units(city)
    built_area = sum(b.width * b.depth for b in city.buildings.values())
    floor_area = sum(f.area_sqm for b in city.buildings.values() for f in b.floors)
    parcel_area = sum(u.area_sqm for _, _, u in units)
    return {
        "buildings": len(city.buildings),
        "floors": sum(len(b.floors) for b in city.buildings.values()),
        "units": len(units),
        "built_area": built_area,
        "floor_area": floor_area,
        "parcel_area": parcel_area,
        "avg_height": float(np.mean([b.height for b in city.buildings.values()])) if city.buildings else 0.0,
    }


def build_cadastral_register(city: CityModel):
    records = []
    for b, f, u in _all_units(city):
        records.append({
            "ULPIN": u.ulpin,
            "Unit ID": u.unit_id,
            "Building ID": b.building_id,
            "Floor ID": f.floor_id,
            "Floor": f.floor_number,
            "Property Type": u.property_type,
            "Owner": u.owner_name,
            "Rights / Tenure": u.rights_type,
            "Area (m²)": round(u.area_sqm, 2),
            "Volume (m³)": round(u.volume_cum, 2),
            "Z Min (m)": round(u.z_min, 2),
            "Z Max (m)": round(u.z_max, 2),
            "LADM Spatial Unit": "LA_SpatialUnit",
        })
    return pd.DataFrame(records)


def building_compliance_snapshot(bld: Building):
    # Prototype indicators, deliberately labelled as screening indicators,
    # not statutory certification.
    footprint = max(0.01, bld.width * bld.depth)
    total_floor_area = footprint * bld.num_floors
    far = total_floor_area / footprint
    coverage = 100.0
    if bld.width > 8 and bld.depth > 8:
        coverage = 86.0
    elif bld.width > 6 and bld.depth > 6:
        coverage = 91.0

    flags = []
    if bld.is_under_construction:
        flags.append("construction-state")
    if bld.height > 24:
        flags.append("height-review")
    if bld.has_awning:
        flags.append("street-interface")
    return far, coverage, flags



# ============================================================================
# 8. SIH26011 — 3D SPATIAL INTELLIGENCE MODULE (ADDITIVE / NON-DESTRUCTIVE)
# ----------------------------------------------------------------------------
# Everything in this section is additive: it reads the CityModel produced by
# the untouched LiDAR / reconstruction pipeline above and derives extra
# analytical layers (ownership status, unauthorized construction, encroachment,
# LiDAR confidence, disasters, time machine, disputes, urban asset layers,
# digital passports, natural-language copilot, spatial search, dashboards).
# None of it mutates PropertyUnit / Floor / Building / CityModel or the
# original render_cesium_viewer geometry logic.
# ============================================================================

UNIT_STATUSES = ["Verified", "Pending", "Disputed", "Government", "Vacant", "Commercial"]

STATUS_COLORS = {
    "Verified": "#10B981",
    "Pending": "#F59E0B",
    "Disputed": "#EF4444",
    "Government": "#6366F1",
    "Vacant": "#64748B",
    "Commercial": "#38BDF8",
}

ASSET_LAYER_OPTIONS = [
    "Solar Panels", "Rooftop Structures", "Shops / Stalls", "Utility Poles"
]

DISASTER_TYPES = ["Flood", "Fire", "Earthquake"]


def _seeded_rng(*parts) -> pyrandom.Random:
    seed = hashlib.md5("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return pyrandom.Random(seed)


# ----------------------------------------------------------------------------
# 8.1 Ownership status classification (Feature 1: 3D Property Ownership Viz)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_unit_status_map(_city: CityModel, city_id: str) -> Dict[str, str]:
    status_map: Dict[str, str] = {}
    for b, f, u in _all_units(_city):
        rng = _seeded_rng(city_id, u.unit_id, "status")
        if "Common" in u.rights_type or "Lobby" in u.property_type or "Corridor" in u.property_type:
            status_map[u.unit_id] = "Government" if rng.random() < 0.3 else "Verified"
        elif "Commercial" in u.property_type:
            status_map[u.unit_id] = "Commercial"
        else:
            roll = rng.random()
            if roll < 0.60:
                status_map[u.unit_id] = "Verified"
            elif roll < 0.76:
                status_map[u.unit_id] = "Pending"
            elif roll < 0.87:
                status_map[u.unit_id] = "Vacant"
            elif roll < 0.96:
                status_map[u.unit_id] = "Disputed"
            else:
                status_map[u.unit_id] = "Government"
    return status_map


def dominant_building_status(b: Building, status_map: Dict[str, str]) -> str:
    counts: Dict[str, int] = {}
    for f in b.floors:
        for u in f.units:
            s = status_map.get(u.unit_id, "Verified")
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "Verified"
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ----------------------------------------------------------------------------
# 8.2 Building-level intelligence: unauthorized construction, encroachment,
#      LiDAR confidence, disaster risk, dispute flags
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_building_intel(_city: CityModel, city_id: str) -> Dict[str, dict]:
    intel: Dict[str, dict] = {}
    for b_id, bld in _city.buildings.items():
        rng = _seeded_rng(city_id, b_id, "intel")

        # --- Unauthorized construction: approved vs LiDAR-derived height/footprint
        is_unauthorized = rng.random() < 0.22
        approved_height = bld.height
        approved_w, approved_d = bld.width, bld.depth
        if is_unauthorized:
            approved_height = bld.height * rng.uniform(0.55, 0.85)
            approved_w = bld.width * rng.uniform(0.85, 0.97)
            approved_d = bld.depth * rng.uniform(0.85, 0.97)
        excess_height = max(0.0, bld.height - approved_height)
        excess_footprint = max(0.0, (bld.width * bld.depth) - (approved_w * approved_d))

        # --- Encroachment: recorded parcel boundary vs actual footprint
        is_encroaching = rng.random() < 0.18
        if is_encroaching:
            parcel_w = bld.width * rng.uniform(0.88, 0.96)
            parcel_d = bld.depth * rng.uniform(0.88, 0.96)
        else:
            parcel_w = bld.width * rng.uniform(1.0, 1.06)
            parcel_d = bld.depth * rng.uniform(1.0, 1.06)
        encroachment_area = max(0.0, (bld.width * bld.depth) - (parcel_w * parcel_d))

        # --- LiDAR confidence / point density
        confidence = rng.uniform(0.55, 0.99)
        point_density = rng.uniform(15.0, 240.0)

        # --- Disaster risk scores
        flood_risk = rng.uniform(0.0, 1.0)
        fire_risk = rng.uniform(0.0, 1.0)
        eq_risk = rng.uniform(0.0, 1.0)

        # --- Cadastral dispute
        is_disputed = rng.random() < 0.14

        intel[b_id] = dict(
            approved_height=approved_height, approved_w=approved_w, approved_d=approved_d,
            is_unauthorized=is_unauthorized, excess_height=excess_height, excess_footprint=excess_footprint,
            parcel_w=parcel_w, parcel_d=parcel_d, is_encroaching=is_encroaching, encroachment_area=encroachment_area,
            confidence=confidence, point_density=point_density,
            flood_risk=flood_risk, fire_risk=fire_risk, eq_risk=eq_risk,
            is_disputed=is_disputed,
        )
    return intel


# ----------------------------------------------------------------------------
# 8.3 Property Time Machine (Feature 5)
# ----------------------------------------------------------------------------
def property_time_machine_states(bld: Building) -> List[dict]:
    rng = _seeded_rng(bld.building_id, "timeline")
    start_year = rng.randint(2005, 2015)
    return [
        {"year": start_year, "label": "Vacant Land", "height_frac": 0.0, "color": "#334155"},
        {"year": start_year + 2, "label": "Foundation & Plinth Laid", "height_frac": 0.06, "color": "#71717A"},
        {"year": start_year + 4, "label": "Under Construction", "height_frac": 0.55, "color": "#E11D48"},
        {"year": start_year + 6, "label": "Structure Completed", "height_frac": 1.0, "color": bld.color_palette["primary"]},
        {"year": 2026, "label": "Verified / Occupied", "height_frac": 1.0, "color": "#10B981"},
    ]


def active_time_machine_state(bld: Building, year: int) -> dict:
    states = property_time_machine_states(bld)
    active = states[0]
    for s in states:
        if year >= s["year"]:
            active = s
    return active


# ----------------------------------------------------------------------------
# 8.4 Construction Change Detection between two (simulated) scan epochs (Feature 7)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def simulate_change_detection(_city: CityModel, city_id: str) -> List[dict]:
    changes = []
    for b_id, bld in _city.buildings.items():
        rng = _seeded_rng(city_id, b_id, "change")
        if rng.random() < 0.16:
            delta_h = rng.uniform(-2.5, 4.5)
            changes.append({
                "building_id": b_id,
                "delta_height": round(delta_h, 2),
                "change_type": "New construction / vertical addition" if delta_h > 0 else "Demolition / height reduction",
                "scan_a_date": "2024-11-02",
                "scan_b_date": "2026-08-14",
            })
    return changes


# ----------------------------------------------------------------------------
# 8.5 Digital Property Passport + QR (Feature 9)
# ----------------------------------------------------------------------------
def generate_property_passport(b: Building, f: Floor, u: PropertyUnit, status: str) -> dict:
    return {
        "ULPIN": u.ulpin,
        "Parcel ID": f"{b.building_id}-PARCEL",
        "Unit ID": u.unit_id,
        "Building ID": b.building_id,
        "Floor": f.floor_number,
        "Area (sqm)": round(u.area_sqm, 2),
        "Usage": u.property_type,
        "Status": status,
        "Owner": u.owner_name,
        "Rights / Tenure": u.rights_type,
        "Verification": ("e-Verified (LADM ISO 19152)" if status == "Verified"
                          else f"{status} — pending field verification"),
        "Issued": datetime.date.today().isoformat(),
    }


def make_qr_code_png_b64(payload: str) -> Optional[str]:
    if not HAS_QRCODE:
        return None
    qr = qrcode.QRCode(border=2, box_size=6)
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#00E5FF", back_color="#0B1018")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------------
# 8.6 Cadastral Copilot — rule-based natural-language query engine (Feature 10)
# ----------------------------------------------------------------------------
def cadastral_copilot_answer(query: str, city: CityModel, status_map: Dict[str, str]) -> str:
    q = query.lower().strip()
    units = _all_units(city)

    def count_status(s):
        return sum(1 for _, _, u in units if status_map.get(u.unit_id) == s)

    intel = compute_building_intel(city, city.city_id)

    if not q:
        return "Ask me something like *'show vacant properties'* or *'which properties have violations?'*"

    if "vacant" in q:
        n = count_status("Vacant")
        ex = [u.unit_id for _, _, u in units if status_map.get(u.unit_id) == "Vacant"][:5]
        return f"There are **{n} vacant** properties in the current dataset. Examples: {', '.join(ex) if ex else '—'}."

    if "violation" in q or "unauthorized" in q or "illegal" in q:
        flagged = [b for b, v in intel.items() if v["is_unauthorized"]]
        return (f"**{len(flagged)} buildings** show suspected unauthorized construction "
                f"(LiDAR-derived height/footprint exceeds the approved plan): "
                f"{', '.join(flagged[:8])}{' …' if len(flagged) > 8 else ''}.")

    if "encroach" in q:
        flagged = [b for b, v in intel.items() if v["is_encroaching"]]
        return (f"**{len(flagged)} buildings** have a footprint crossing the recorded parcel boundary: "
                f"{', '.join(flagged[:8])}{' …' if len(flagged) > 8 else ''}.")

    if "commercial" in q:
        n = count_status("Commercial")
        return f"There are **{n} commercial units** in the current city model."

    if "dispute" in q:
        n = count_status("Disputed")
        b_n = sum(1 for v in intel.values() if v["is_disputed"])
        return f"**{n} units** and **{b_n} buildings** are currently flagged under ownership / boundary dispute."

    if "government" in q:
        n = count_status("Government")
        return f"**{n} units** are classified as Government-owned or common-area property."

    if "verified" in q:
        n = count_status("Verified")
        return f"**{n} units** are fully e-Verified against the registry."

    if "pending" in q:
        n = count_status("Pending")
        return f"**{n} units** currently have a Pending verification status."

    if "how many building" in q or "total building" in q or "number of building" in q:
        return f"The current city model contains **{len(city.buildings)} buildings**."

    if "average floor" in q or "avg floor" in q or "average height" in q:
        avg = float(np.mean([b.num_floors for b in city.buildings.values()])) if city.buildings else 0.0
        return f"The average building height is **{avg:.1f} floors**."

    if "confidence" in q or "point density" in q or "lidar quality" in q:
        low_conf = [b for b, v in intel.items() if v["confidence"] < 0.68]
        return (f"**{len(low_conf)} buildings** have low LiDAR reconstruction confidence (<0.68) and may "
                f"need a re-survey: {', '.join(low_conf[:8])}{' …' if len(low_conf) > 8 else ''}.")

    if "flood" in q:
        at_risk = [b for b, v in intel.items() if v["flood_risk"] > 0.65]
        return f"**{len(at_risk)} buildings** fall in the high flood-risk simulation zone."

    if "fire" in q:
        at_risk = [b for b, v in intel.items() if v["fire_risk"] > 0.65]
        return f"**{len(at_risk)} buildings** fall in the high fire-risk simulation zone."

    if "earthquake" in q or "seismic" in q:
        at_risk = [b for b, v in intel.items() if v["eq_risk"] > 0.65]
        return f"**{len(at_risk)} buildings** fall in the high seismic-risk simulation zone."

    if "unit" in q and ("total" in q or "how many" in q):
        return f"There are **{len(units)} registered 3D property units** across {len(city.buildings)} buildings."

    return ("I can answer things like *'show vacant properties'*, *'which properties have violations?'*, "
            "*'how many commercial units are here?'*, *'disputed properties'*, *'encroaching buildings'*, "
            "*'flood risk'* or *'LiDAR confidence'*. Try rephrasing your query.")


# ----------------------------------------------------------------------------
# 8.7 3D Spatial Search (Feature 11)
# ----------------------------------------------------------------------------
def spatial_search(city: CityModel, status_map: Dict[str, str], query: str = "",
                    usage: str = "Any", status_filter: str = "Any",
                    min_area: float = 0.0, max_area: float = 100000.0,
                    floor: Optional[int] = None):
    results = []
    q = query.strip().lower()
    for b, f, u in _all_units(city):
        if q:
            hay = " ".join([u.ulpin, u.unit_id, b.building_id, f.floor_id, u.owner_name, u.property_type]).lower()
            if q not in hay:
                continue
        if usage != "Any" and usage.lower() not in u.property_type.lower():
            continue
        s = status_map.get(u.unit_id, "Verified")
        if status_filter != "Any" and s != status_filter:
            continue
        if not (min_area <= u.area_sqm <= max_area):
            continue
        if floor is not None and f.floor_number != floor:
            continue
        results.append((b, f, u, s))
    return results


# ----------------------------------------------------------------------------
# 8.8 Urban Intelligence Dashboard + FAR / Land Utilization Analysis (13/14)
# ----------------------------------------------------------------------------
def urban_intelligence_summary(city: CityModel, status_map: Dict[str, str], intel: Dict[str, dict]) -> dict:
    units = _all_units(city)
    footprint_total = sum(b.width * b.depth for b in city.buildings.values())
    floor_area_total = sum(f.area_sqm for b in city.buildings.values() for f in b.floors)
    site_area = 175.0 * 150.0  # matches benchmark road/site extents used across the module
    far_values = [ (b.width * b.depth * b.num_floors) / max(0.01, b.width * b.depth) for b in city.buildings.values() ]
    return {
        "buildings": len(city.buildings),
        "units": len(units),
        "avg_floors": float(np.mean([b.num_floors for b in city.buildings.values()])) if city.buildings else 0.0,
        "built_footprint": footprint_total,
        "total_floor_area": floor_area_total,
        "site_area": site_area,
        "open_space_pct": max(0.0, 100.0 * (1.0 - footprint_total / site_area)),
        "avg_far": float(np.mean(far_values)) if far_values else 0.0,
        "vacant_units": sum(1 for s in status_map.values() if s == "Vacant"),
        "disputed_units": sum(1 for s in status_map.values() if s == "Disputed"),
        "commercial_units": sum(1 for s in status_map.values() if s == "Commercial"),
        "government_units": sum(1 for s in status_map.values() if s == "Government"),
        "verified_units": sum(1 for s in status_map.values() if s == "Verified"),
        "pending_units": sum(1 for s in status_map.values() if s == "Pending"),
        "unauthorized_buildings": sum(1 for v in intel.values() if v["is_unauthorized"]),
        "encroaching_buildings": sum(1 for v in intel.values() if v["is_encroaching"]),
        "disputed_buildings": sum(1 for v in intel.values() if v["is_disputed"]),
        "avg_confidence": float(np.mean([v["confidence"] for v in intel.values()])) if intel else 0.0,
    }


def far_utilization_table(city: CityModel, intel: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for b_id, bld in city.buildings.items():
        footprint = bld.width * bld.depth
        floor_area = footprint * bld.num_floors
        far = floor_area / max(0.01, footprint)
        v = intel[b_id]
        rows.append({
            "Building": b_id,
            "Footprint (m²)": round(footprint, 1),
            "Floor Area (m²)": round(floor_area, 1),
            "FAR": round(far, 2),
            "Floors": bld.num_floors,
            "Height (m)": round(bld.height, 1),
            "Approved Height (m)": round(v["approved_height"], 1),
            "Excess Height (m)": round(v["excess_height"], 1),
            "Encroachment (m²)": round(v["encroachment_area"], 1),
            "LiDAR Confidence": round(v["confidence"], 2),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 8.9 3D overlay builders — merged into the Cesium viewer as generic
#      translucent boxes / points, layered on top of the untouched pipeline.
# ----------------------------------------------------------------------------
def build_unauthorized_overlays(city: CityModel, intel: Dict[str, dict]) -> List[dict]:
    overlays = []
    for b_id, bld in city.buildings.items():
        v = intel[b_id]
        if not v["is_unauthorized"]:
            continue
        lon, lat = local_to_wgs84(bld.cx, bld.cy, city.ref_lon, city.ref_lat)
        overlays.append({
            "id": f"{b_id}-UNAUTH", "name": f"Suspected Extra Construction: {b_id}",
            "lon": lon, "lat": lat, "w": bld.width * 0.55, "d": bld.depth * 0.55,
            "base_z": v["approved_height"], "h": max(0.5, v["excess_height"]),
            "color": "#F97316", "alpha": 0.72, "outline_color": "#FDBA74",
            "metadata": {
                "Status": "Suspected unauthorized construction",
                "Approved Height": f"{v['approved_height']:.1f} m",
                "LiDAR Height": f"{bld.height:.1f} m",
                "Excess Height": f"{v['excess_height']:.1f} m",
            }
        })
    return overlays


def build_encroachment_overlays(city: CityModel, intel: Dict[str, dict]) -> List[dict]:
    overlays = []
    for b_id, bld in city.buildings.items():
        v = intel[b_id]
        if not v["is_encroaching"]:
            continue
        lon, lat = local_to_wgs84(bld.cx, bld.cy, city.ref_lon, city.ref_lat)
        overlays.append({
            "id": f"{b_id}-ENCR", "name": f"Parcel Encroachment: {b_id}",
            "lon": lon, "lat": lat, "w": bld.width + 1.0, "d": bld.depth + 1.0,
            "base_z": 0.03, "h": 0.3, "color": "#EF4444", "alpha": 0.55,
            "outline_color": "#FCA5A5",
            "metadata": {
                "Status": "Crosses recorded parcel boundary",
                "Encroachment Area": f"{v['encroachment_area']:.1f} m²",
                "Recorded Parcel": f"{v['parcel_w']:.1f} m x {v['parcel_d']:.1f} m",
            }
        })
    return overlays


def build_confidence_heatmap_points(city: CityModel, intel: Dict[str, dict]) -> List[dict]:
    points = []
    for b_id, bld in city.buildings.items():
        v = intel[b_id]
        lon, lat = local_to_wgs84(bld.cx, bld.cy, city.ref_lon, city.ref_lat)
        conf = v["confidence"]
        color = "#10B981" if conf > 0.85 else ("#F59E0B" if conf > 0.68 else "#EF4444")
        points.append({
            "lon": lon, "lat": lat, "z": bld.height + 2.0,
            "radius": max(bld.width, bld.depth) * 0.6,
            "color": color, "alpha": 0.5,
        })
    return points


def build_disaster_overlays(city: CityModel, intel: Dict[str, dict], hazard: str) -> List[dict]:
    key = {"Flood": "flood_risk", "Fire": "fire_risk", "Earthquake": "eq_risk"}[hazard]
    overlays = []
    for b_id, bld in city.buildings.items():
        risk = intel[b_id][key]
        if risk < 0.5:
            continue
        lon, lat = local_to_wgs84(bld.cx, bld.cy, city.ref_lon, city.ref_lat)
        color = "#EF4444" if risk > 0.8 else "#F59E0B"
        h = (1.2 * risk) if hazard == "Flood" else 1.0
        base_z = 0.0 if hazard == "Flood" else bld.height + 0.4
        overlays.append({
            "id": f"{b_id}-HAZ-{hazard}", "name": f"{hazard} Risk: {b_id}",
            "lon": lon, "lat": lat, "w": bld.width + 0.6, "d": bld.depth + 0.6,
            "base_z": base_z, "h": max(0.3, h), "color": color,
            "alpha": 0.45 if hazard == "Flood" else 0.6, "outline_color": color,
            "metadata": {"Hazard": hazard, "Risk Score": f"{risk:.2f}",
                         "Affected": "Yes" if risk > 0.65 else "Monitor"}
        })
    return overlays


def build_underground_overlays(city: CityModel) -> List[dict]:
    overlays = []
    utility_specs = [
        ("Water Line", -1.2, "#38BDF8"), ("Sewer Line", -2.0, "#84CC16"),
        ("Electrical Duct", -1.6, "#F59E0B"), ("Gas Line", -1.8, "#F97316"),
    ]
    x_coords = city.road_x_coords if city.road_x_coords else [
        b.cx for i, b in enumerate(city.buildings.values()) if i < 3
    ]
    for xr in x_coords:
        for label, depth, color in utility_specs:
            lon, lat = local_to_wgs84(xr, 70.0, city.ref_lon, city.ref_lat)
            overlays.append({
                "id": f"UG-{xr}-{label}", "name": label,
                "lon": lon, "lat": lat, "w": 0.6, "d": 150.0,
                "base_z": depth, "h": 0.4, "color": color, "alpha": 0.8,
                "outline_color": color,
                "metadata": {"Type": "Underground Utility", "Layer": label, "Depth": f"{abs(depth):.1f} m"}
            })
    for b_id, bld in city.buildings.items():
        rng = _seeded_rng(b_id, "basement")
        if rng.random() < 0.32:
            lon, lat = local_to_wgs84(bld.cx, bld.cy, city.ref_lon, city.ref_lat)
            overlays.append({
                "id": f"{b_id}-BASEMENT", "name": f"Basement: {b_id}",
                "lon": lon, "lat": lat, "w": bld.width * 0.9, "d": bld.depth * 0.9,
                "base_z": -2.8, "h": 2.6, "color": "#334155", "alpha": 0.9,
                "outline_color": "#64748B", "metadata": {"Type": "Basement Level", "Building": b_id}
            })
    return overlays


def build_selected_building_underground_slabs(bld: Building, city: CityModel) -> List[dict]:
    """
    Real, controllable underground/basement floor slabs for ONE building
    (as opposed to build_underground_overlays(), which only sprinkles
    generic utility lines + a probabilistic basement box across the city).
    Uses bld.num_underground_floors, which is editable from the
    "Structure Editor" workspace. Floor height mirrors the ground-floor
    height of the building so basement levels look proportionate.
    """
    overlays = []
    n = max(0, int(getattr(bld, "num_underground_floors", 0)))
    if n <= 0 or not bld.floors:
        return overlays
    floor_h = bld.floors[0].height
    lon, lat = local_to_wgs84(bld.cx, bld.cy, city.ref_lon, city.ref_lat)
    for i in range(1, n + 1):
        base_z = -i * floor_h
        level_name = f"B{i}"
        overlays.append({
            "id": f"{bld.building_id}-UGF-{level_name}",
            "name": f"{bld.building_id} — Underground Level {level_name}",
            "lon": lon, "lat": lat,
            "w": bld.width * 0.96, "d": bld.depth * 0.96,
            "base_z": base_z, "h": floor_h * 0.94,
            "color": "#1E1B4B", "alpha": 0.85, "outline_color": "#818CF8",
            "metadata": {
                "Type": "Underground / Basement Floor",
                "Level": level_name,
                "Building": bld.building_id,
                "Depth Below Grade": f"{abs(base_z):.1f} m",
            }
        })
    return overlays


def convert_floorplan_image_to_wall_overlays(image_bytes: bytes, bld: "Building", flr: "Floor",
                                              city: CityModel, wall_height: float = 2.6,
                                              max_walls: int = 140) -> Tuple[List[dict], int]:
    """
    Heuristic 2D floor-plan (raster image) -> 3D wall extrusion.
    This is a lightweight computer-vision approximation (edge/contour based),
    not a true CAD vectorization: it detects dark line-work in the uploaded
    image, treats each detected segment as a wall, and extrudes it to
    wall_height at the target floor's elevation. Rotated walls are
    approximated as axis-aligned boxes (the underlying 3D viewer renders
    unrotated boxes), so diagonal walls will look "stepped" rather than
    perfectly angled. Good enough for a quick massing check; not a
    substitute for real CAD import.
    Returns (overlay_boxes, wall_count).
    """
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image — please upload a PNG or JPG floor plan.")

    # Downscale very large scans for performance.
    max_dim = 900
    h0, w0 = img.shape[:2]
    if max(h0, w0) > max_dim:
        scale = max_dim / max(h0, w0)
        img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)

    img_h, img_w = img.shape[:2]

    # Otsu threshold; auto-detect polarity (walls should end up as the
    # minority "foreground" class in a typical white-background floor plan).
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(bin_img) > bin_img.size * 0.5:
        bin_img = cv2.bitwise_not(bin_img)

    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(6.0, (img_w * img_h) * 0.00006)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        candidates.append((area, x, y, w, h))

    # Keep the largest wall-like segments so the scene stays fast.
    candidates.sort(key=lambda t: -t[0])
    candidates = candidates[:max_walls]

    scale_x = bld.width / img_w
    scale_y = bld.depth / img_h
    base_z = flr.elevation

    overlays = []
    for idx, (area, x, y, w, h) in enumerate(candidates):
        local_cx = (x + w / 2.0 - img_w / 2.0) * scale_x
        # Image row 0 is the top; flip so it maps to +depth in local space.
        local_cy = (img_h / 2.0 - (y + h / 2.0)) * scale_y
        wall_w = max(0.12, w * scale_x)
        wall_d = max(0.12, h * scale_y)
        lon, lat = local_to_wgs84(bld.cx + local_cx, bld.cy + local_cy, city.ref_lon, city.ref_lat)
        overlays.append({
            "id": f"{flr.floor_id}-2D3D-WALL-{idx}",
            "name": f"Imported Wall {idx + 1}",
            "lon": lon, "lat": lat,
            "w": wall_w, "d": wall_d,
            "base_z": base_z, "h": wall_height,
            "color": "#E2E8F0", "alpha": 0.96, "outline_color": "#94A3B8",
            "metadata": {"Type": "Imported 2D→3D Wall", "Source Floor": flr.floor_id, "Building": bld.building_id}
        })
    return overlays, len(overlays)


_DXF_LINE_RE = None  # placeholder kept for readability; parsing is done procedurally below


def parse_dxf_walls_to_overlays(dxf_text: str, bld: "Building", flr: "Floor",
                                 city: CityModel, wall_height: float = 2.6,
                                 max_walls: int = 300) -> Tuple[List[dict], int]:
    """
    Minimal, dependency-free DXF (ASCII, R12-style group-code) reader for
    "AutoCAD map" wall input. Reads LINE entities (group 10/20 = start x/y,
    11/21 = end x/y) and LWPOLYLINE / POLYLINE vertices (group 10/20 per
    vertex), and turns each resulting segment into an extruded wall box.
    This intentionally only understands plain line geometry (the common
    case for wall centerlines exported from AutoCAD) — blocks, arcs,
    splines and layers/xrefs are ignored rather than guessed at.
    Coordinates in the DXF are assumed to be in the same units as the
    building footprint (meters) and are re-centered on the building.
    Returns (overlay_boxes, segment_count).
    """
    lines = dxf_text.replace("\r\n", "\n").splitlines()
    tokens = []
    i = 0
    while i + 1 < len(lines):
        code = lines[i].strip()
        value = lines[i + 1].strip()
        tokens.append((code, value))
        i += 2

    segments: List[Tuple[float, float, float, float]] = []  # x1,y1,x2,y2 in DXF space

    idx = 0
    n = len(tokens)
    while idx < n:
        code, value = tokens[idx]
        if code == "0" and value == "LINE":
            pts = {}
            j = idx + 1
            while j < n and not (tokens[j][0] == "0"):
                c, v = tokens[j]
                if c in ("10", "20", "11", "21"):
                    try:
                        pts[c] = float(v)
                    except ValueError:
                        pass
                j += 1
            if all(k in pts for k in ("10", "20", "11", "21")):
                segments.append((pts["10"], pts["20"], pts["11"], pts["21"]))
            idx = j
            continue
        if code == "0" and value in ("LWPOLYLINE", "POLYLINE"):
            j = idx + 1
            verts = []
            cur = {}
            closed = False
            while j < n and not (tokens[j][0] == "0" and tokens[j][1] not in ("VERTEX",)):
                c, v = tokens[j]
                if c == "70":
                    try:
                        closed = (int(float(v)) & 1) == 1
                    except ValueError:
                        pass
                if c == "10":
                    if cur:
                        verts.append(cur)
                    cur = {"x": None, "y": None}
                    try:
                        cur["x"] = float(v)
                    except ValueError:
                        pass
                if c == "20" and cur:
                    try:
                        cur["y"] = float(v)
                    except ValueError:
                        pass
                j += 1
            if cur.get("x") is not None:
                verts.append(cur)
            verts = [v for v in verts if v.get("x") is not None and v.get("y") is not None]
            for k in range(len(verts) - 1):
                segments.append((verts[k]["x"], verts[k]["y"], verts[k + 1]["x"], verts[k + 1]["y"]))
            if closed and len(verts) > 2:
                segments.append((verts[-1]["x"], verts[-1]["y"], verts[0]["x"], verts[0]["y"]))
            idx = j
            continue
        idx += 1

    if not segments:
        return [], 0

    xs = [p for s in segments for p in (s[0], s[2])]
    ys = [p for s in segments for p in (s[1], s[3])]
    dxf_cx = (min(xs) + max(xs)) / 2.0
    dxf_cy = (min(ys) + max(ys)) / 2.0
    dxf_w = max(0.01, max(xs) - min(xs))
    dxf_d = max(0.01, max(ys) - min(ys))
    # Fit the drawing's extents into the building's footprint so a plan
    # drawn at real-world scale still lands on the right floor.
    fit_scale = min(bld.width / dxf_w, bld.depth / dxf_d, 1.0) if (dxf_w > 0 and dxf_d > 0) else 1.0

    base_z = flr.elevation
    overlays = []
    segments = segments[:max_walls]
    for idx2, (x1, y1, x2, y2) in enumerate(segments):
        mx = ((x1 + x2) / 2.0 - dxf_cx) * fit_scale
        my = ((y1 + y2) / 2.0 - dxf_cy) * fit_scale
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 * fit_scale
        # Approximate as axis-aligned: long axis follows whichever of dx/dy
        # dominates (the 3D box renderer here only supports axis-aligned
        # boxes, so true diagonal walls are approximated this way).
        is_horizontal = abs(x2 - x1) >= abs(y2 - y1)
        wall_w = max(0.15, length) if is_horizontal else 0.2
        wall_d = 0.2 if is_horizontal else max(0.15, length)
        lon, lat = local_to_wgs84(bld.cx + mx, bld.cy + my, city.ref_lon, city.ref_lat)
        overlays.append({
            "id": f"{flr.floor_id}-DXF-WALL-{idx2}",
            "name": f"CAD Wall {idx2 + 1}",
            "lon": lon, "lat": lat,
            "w": wall_w, "d": wall_d,
            "base_z": base_z, "h": wall_height,
            "color": "#FBBF24", "alpha": 0.96, "outline_color": "#78350F",
            "metadata": {"Type": "Imported CAD (DXF) Wall", "Source Floor": flr.floor_id, "Building": bld.building_id}
        })
    return overlays, len(overlays)


def resize_building_floors(bld: "Building", new_count: int, names: Optional[List[str]] = None) -> None:
    """
    Frontend structural editing: add or remove floors on an existing
    building in place, keeping elevations/units consistent. New floors
    reuse the building's existing ground-floor height and layout logic;
    removed floors are trimmed from the top down.
    """
    fallback_names = names or ["Registered Owner", "Property Owner", "Owners Association"]
    new_count = max(1, min(int(new_count), 80))
    current = len(bld.floors)
    if new_count == current:
        bld.num_floors = new_count
        return

    if new_count < current:
        bld.floors = bld.floors[:new_count]
    else:
        floor_h = bld.floors[-1].height if bld.floors else (bld.height / max(1, current))
        next_elev = (bld.floors[-1].elevation + bld.floors[-1].height) if bld.floors else 0.0
        for i in range(current, new_count):
            floor_num = i + 1
            f_id = f"{bld.building_id}-FLR-EXT{floor_num:02d}"
            units = generate_architectural_floorplan(
                bld.cx, bld.cy, bld.width, bld.depth, bld.building_id, f_id,
                floor_num, next_elev, floor_h, fallback_names
            )
            bld.floors.append(Floor(
                floor_id=f_id, building_id=bld.building_id, floor_number=floor_num,
                elevation=round(next_elev, 2), abs_elevation=round(next_elev, 2),
                height=floor_h, area_sqm=round(bld.width * bld.depth, 2), units=units,
            ))
            next_elev += floor_h

    bld.num_floors = new_count
    bld.height = round(sum(f.height for f in bld.floors), 2)


def export_building_to_dxf(bld: "Building") -> bytes:
    """
    Dependency-free ASCII DXF (R12 group-code format) export of a
    building's floor footprints and unit boundaries as 3D polylines, one
    layer per floor. DXF (not binary DWG) is used deliberately: AutoCAD,
    BricsCAD, and every major CAD package open DXF natively via File > Open
    / Import, and producing it requires no proprietary Autodesk SDK —
    binary .dwg is a closed format Autodesk does not license for
    third-party writers.
    """
    def esc(s: str) -> str:
        return str(s).replace("\n", " ")[:255]

    out = []
    out += ["0", "SECTION", "2", "HEADER", "0", "ENDSEC"]

    out += ["0", "SECTION", "2", "TABLES"]
    out += ["0", "TABLE", "2", "LAYER", "70", str(len(bld.floors) + 1)]
    out += ["0", "LAYER", "2", "BUILDING_OUTLINE", "70", "0", "62", "7", "6", "CONTINUOUS"]
    for f in bld.floors:
        out += ["0", "LAYER", "2", esc(f.floor_id), "70", "0", "62", "5", "6", "CONTINUOUS"]
    out += ["0", "ENDTAB", "0", "ENDSEC"]

    out += ["0", "SECTION", "2", "ENTITIES"]

    def add_3d_polyline(layer: str, pts: List[Tuple[float, float, float]], closed: bool = True):
        out.extend(["0", "POLYLINE", "8", esc(layer), "66", "1", "70", "8"])
        for (x, y, z) in pts:
            out.extend(["0", "VERTEX", "8", esc(layer), "10", f"{x:.3f}", "20", f"{y:.3f}", "30", f"{z:.3f}", "70", "32"])
        if closed and pts:
            x, y, z = pts[0]
            out.extend(["0", "VERTEX", "8", esc(layer), "10", f"{x:.3f}", "20", f"{y:.3f}", "30", f"{z:.3f}", "70", "32"])
        out.extend(["0", "SEQEND"])

    hw, hd = bld.width / 2.0, bld.depth / 2.0
    for f in bld.floors:
        z = f.elevation
        footprint = [
            (bld.cx - hw, bld.cy - hd, z), (bld.cx + hw, bld.cy - hd, z),
            (bld.cx + hw, bld.cy + hd, z), (bld.cx - hw, bld.cy + hd, z),
        ]
        add_3d_polyline("BUILDING_OUTLINE", footprint)
        for u in f.units:
            uw, ud = u.w / 2.0, u.d / 2.0
            unit_pts = [
                (u.cx - uw, u.cy - ud, z), (u.cx + uw, u.cy - ud, z),
                (u.cx + uw, u.cy + ud, z), (u.cx - uw, u.cy + ud, z),
            ]
            add_3d_polyline(f.floor_id, unit_pts)

    # Roof cap
    top_z = bld.floors[-1].elevation + bld.floors[-1].height if bld.floors else bld.height
    add_3d_polyline("BUILDING_OUTLINE", [
        (bld.cx - hw, bld.cy - hd, top_z), (bld.cx + hw, bld.cy - hd, top_z),
        (bld.cx + hw, bld.cy + hd, top_z), (bld.cx - hw, bld.cy + hd, top_z),
    ])

    out += ["0", "ENDSEC", "0", "EOF"]
    return ("\n".join(out) + "\n").encode("ascii", errors="replace")


def build_dispute_overlays(city: CityModel, intel: Dict[str, dict]) -> List[dict]:
    overlays = []
    for b_id, bld in city.buildings.items():
        if not intel[b_id]["is_disputed"]:
            continue
        rng = _seeded_rng(b_id, "dispute-claim")
        ox, oy = rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6)
        lon_a, lat_a = local_to_wgs84(bld.cx, bld.cy, city.ref_lon, city.ref_lat)
        lon_b, lat_b = local_to_wgs84(bld.cx + ox, bld.cy + oy, city.ref_lon, city.ref_lat)
        overlays.append({
            "id": f"{b_id}-CLAIM-A", "name": f"Claim A: {b_id}",
            "lon": lon_a, "lat": lat_a, "w": bld.width + 0.4, "d": bld.depth + 0.4,
            "base_z": 0.06, "h": 0.3, "color": "#6366F1", "alpha": 0.4, "outline_color": "#A5B4FC",
            "metadata": {"Claimant": "Registered Owner", "Review Status": "Under Review"}
        })
        overlays.append({
            "id": f"{b_id}-CLAIM-B", "name": f"Claim B: {b_id}",
            "lon": lon_b, "lat": lat_b, "w": bld.width * 0.85, "d": bld.depth * 0.85,
            "base_z": 0.06, "h": 0.3, "color": "#EC4899", "alpha": 0.4, "outline_color": "#F9A8D4",
            "metadata": {"Claimant": "Competing Claimant", "Review Status": "Under Review"}
        })
    return overlays


def build_change_detection_overlays(city: CityModel, changes: List[dict]) -> List[dict]:
    overlays = []
    for c in changes:
        bld = city.buildings.get(c["building_id"])
        if not bld:
            continue
        lon, lat = local_to_wgs84(bld.cx, bld.cy, city.ref_lon, city.ref_lat)
        color = "#22D3EE" if c["delta_height"] > 0 else "#F87171"
        overlays.append({
            "id": f"{c['building_id']}-CHG", "name": f"Change Detected: {c['building_id']}",
            "lon": lon, "lat": lat, "w": bld.width * 0.45, "d": bld.depth * 0.45,
            "base_z": bld.height, "h": max(0.6, abs(c["delta_height"])),
            "color": color, "alpha": 0.8, "outline_color": color,
            "metadata": {"Delta Height": f"{c['delta_height']:+.1f} m", "Change Type": c["change_type"],
                         "Scan A": c["scan_a_date"], "Scan B": c["scan_b_date"]}
        })
    return overlays


def build_time_machine_overlay(bld: Building, city: CityModel, year: int) -> Tuple[List[dict], dict]:
    active = active_time_machine_state(bld, year)
    lon, lat = local_to_wgs84(bld.cx, bld.cy, city.ref_lon, city.ref_lat)
    h = max(0.3, bld.height * active["height_frac"])
    overlay = [{
        "id": f"{bld.building_id}-TM", "name": f"{bld.building_id} @ {year}",
        "lon": lon, "lat": lat, "w": bld.width, "d": bld.depth,
        "base_z": 0.0, "h": h, "color": active["color"], "alpha": 0.92, "outline_color": "#F8FAFC",
        "metadata": {"Year": str(year), "State": active["label"]}
    }]
    return overlay, active


def build_urban_asset_layer_overlays(city: CityModel, active_layers: List[str]) -> List[dict]:
    overlays = []
    ref_lon, ref_lat = city.ref_lon, city.ref_lat

    if "Solar Panels" in active_layers:
        for b_id, bld in city.buildings.items():
            rng = _seeded_rng(b_id, "solar")
            if rng.random() < 0.4:
                lon, lat = local_to_wgs84(bld.cx, bld.cy, ref_lon, ref_lat)
                overlays.append({
                    "id": f"{b_id}-SOLAR", "name": "Rooftop Solar Array",
                    "lon": lon, "lat": lat, "w": bld.width * 0.5, "d": bld.depth * 0.5,
                    "base_z": bld.height + 0.05, "h": 0.15, "color": "#1D4ED8", "alpha": 0.9,
                    "outline_color": "#93C5FD", "metadata": {"Asset": "Solar Panel Array"}
                })

    if "Rooftop Structures" in active_layers:
        for b_id, bld in city.buildings.items():
            rng = _seeded_rng(b_id, "roofstruct")
            if rng.random() < 0.3:
                lon, lat = local_to_wgs84(bld.cx - bld.width * 0.2, bld.cy, ref_lon, ref_lat)
                overlays.append({
                    "id": f"{b_id}-ROOFSTR", "name": "Rooftop Structure",
                    "lon": lon, "lat": lat, "w": 1.6, "d": 1.6,
                    "base_z": bld.height + 0.05, "h": 1.4, "color": "#78716C", "alpha": 0.95,
                    "outline_color": "#D6D3D1", "metadata": {"Asset": "Rooftop Structure (survey flag)"}
                })

    if "Shops / Stalls" in active_layers:
        coords = [(35.5, 42.5), (74.6, 50.5), (114.6, 48.5), (74.9, 110.5), (41.4, 55.0)]
        for i, (sx, sy) in enumerate(coords):
            lon, lat = local_to_wgs84(sx, sy, ref_lon, ref_lat)
            overlays.append({
                "id": f"STALL-{i}", "name": "Street Vendor Stall",
                "lon": lon, "lat": lat, "w": 1.4, "d": 1.4,
                "base_z": 0.08, "h": 1.0, "color": "#DC2626", "alpha": 0.9,
                "outline_color": "#FCA5A5", "metadata": {"Asset": "Food Stall / Thela"}
            })

    if "Utility Poles" in active_layers:
        x_coords = city.road_x_coords if city.road_x_coords else []
        for i, xr in enumerate(x_coords):
            lon, lat = local_to_wgs84(xr + 3.5, 20.0 + i * 15, ref_lon, ref_lat)
            overlays.append({
                "id": f"POLE-{i}", "name": "Utility Pole",
                "lon": lon, "lat": lat, "w": 0.3, "d": 0.3,
                "base_z": 0.1, "h": 6.0, "color": "#A8A29E", "alpha": 0.95,
                "outline_color": "#E7E5E4", "metadata": {"Asset": "Electrical Utility Pole"}
            })

    return overlays


def reset_selection():
    st.session_state.current_level = "CITY"
    st.session_state.selected_building_id = None
    st.session_state.selected_floor_id = None
    st.session_state.selected_unit_id = None


# ============================================================================
# 7. STREAMLIT APPLICATION CONTROLLER
# ============================================================================
st.set_page_config(
    page_title="Indian 3D Cadastral Twin | SIH26011",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.stApp { background-color: #070B14; color: #F8FAFC; }
.breadcrumb-bar {
    font-family: monospace; font-size: 1.05rem; background: #111827;
    padding: 10px 18px; border-radius: 8px; border-left: 4px solid #00F0FF;
    margin-bottom: 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
}
.metric-card {
    background: #111827; border: 1px solid #263244; padding: 14px;
    border-radius: 5px; margin-bottom: 12px;
}
[data-testid="stSidebar"] {
    background: #0B1018;
    border-right: 1px solid #263244;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    letter-spacing: 0.08em;
}
div[data-testid="stMetric"] {
    background: #0B1018;
    border: 1px solid #263244;
    padding: 8px 10px;
    border-radius: 4px;
}

</style>
""", unsafe_allow_html=True)

if "city_model" not in st.session_state:
    st.session_state.city_model = generate_synthetic_city()
if "current_level" not in st.session_state:
    st.session_state.current_level = "CITY"
if "selected_building_id" not in st.session_state:
    st.session_state.selected_building_id = None
if "selected_floor_id" not in st.session_state:
    st.session_state.selected_floor_id = None
if "selected_unit_id" not in st.session_state:
    st.session_state.selected_unit_id = None
if "explode_floors" not in st.session_state:
    st.session_state.explode_floors = False
if "cutaway_mode" not in st.session_state:
    st.session_state.cutaway_mode = False
if "render_mode" not in st.session_state:
    st.session_state.render_mode = "REALISTIC"
if "show_ground" not in st.session_state:
    st.session_state.show_ground = True

if "show_vegetation" not in st.session_state:
    st.session_state.show_vegetation = True
if "workspace_mode" not in st.session_state:
    st.session_state.workspace_mode = "Survey"
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "show_register" not in st.session_state:
    st.session_state.show_register = False
if "show_compliance" not in st.session_state:
    st.session_state.show_compliance = False
if "show_provenance" not in st.session_state:
    st.session_state.show_provenance = False

# --- Section 8 additive UI state (Spatial Intelligence Module) ---
if "layer_ownership_status" not in st.session_state:
    st.session_state.layer_ownership_status = False
if "layer_unauthorized" not in st.session_state:
    st.session_state.layer_unauthorized = False
if "layer_encroachment" not in st.session_state:
    st.session_state.layer_encroachment = False
if "layer_confidence" not in st.session_state:
    st.session_state.layer_confidence = False
if "layer_underground" not in st.session_state:
    st.session_state.layer_underground = False
if "layer_disputes" not in st.session_state:
    st.session_state.layer_disputes = False
if "layer_change_detection" not in st.session_state:
    st.session_state.layer_change_detection = False
if "active_asset_layers" not in st.session_state:
    st.session_state.active_asset_layers = []
if "disaster_mode" not in st.session_state:
    st.session_state.disaster_mode = "None"
if "time_machine_active" not in st.session_state:
    st.session_state.time_machine_active = False
if "time_machine_year" not in st.session_state:
    st.session_state.time_machine_year = 2026

# --- Proposed-enhancement UI state (additive) ---
if "show_underground_floors" not in st.session_state:
    st.session_state.show_underground_floors = False
if "cad_import_overlays" not in st.session_state:
    # {building_id: {floor_id: [overlay_box, ...]}}
    st.session_state.cad_import_overlays = {}


with st.sidebar:
    # Operational field-console style instead of a generic AI dashboard.
    st.markdown("## FIELD CONSOLE")
    st.caption("3D CADASTRAL / LAND ADMINISTRATION")
    st.markdown("**CASE:** SIH26011  •  **REV:** 01")
    st.markdown("---")

    WORKSPACE_MODES = [
        "Survey", "Register", "Compliance", "Provenance",
        "Ownership 3D", "Violations", "Intelligence Dashboard", "FAR / Utilization",
        "Cadastral Copilot", "Spatial Search", "Digital Passport",
        "Time Machine", "Disaster Simulation", "Change Detection", "Dispute Mode",
        "2D→3D / CAD Import", "Structure Editor", "Analytics Dashboard", "AutoCAD Export",
    ]
    st.selectbox(
        "WORKSPACE",
        WORKSPACE_MODES,
        key="workspace_mode"
    )

    st.markdown("---")
    st.markdown("### VIEW CONTROL")
    RENDER_MODES = ["REALISTIC", "LIDAR_ELEVATION", "CADASTRAL_LADM", "STRUCTURAL_XRAY", "DENSITY_FAR", "OWNERSHIP_STATUS"]
    st.selectbox(
        "Representation",
        RENDER_MODES,
        key="render_mode"
    )

    c1, c2 = st.columns(2)
    with c1:
        st.toggle("GROUND", key="show_ground")
    with c2:
        st.toggle("VEGETATION", key="show_vegetation")

    st.toggle(
        "EXPLODE FLOOR STRATA", key="explode_floors",
        help="Vertical Property Stack: physically separates floors for City → Building → Floor → Unit inspection."
    )
    st.toggle("CUTAWAY SPATIAL UNITS", key="cutaway_mode")

    st.markdown("---")
    st.markdown("### 3D ANALYTICAL LAYERS")
    st.caption("Toggle detection & simulation overlays on the 3D viewport.")
    st.caption("Tip: set Representation = OWNERSHIP_STATUS above to color every unit by status.")
    l1, l2 = st.columns(2)
    with l1:
        st.toggle("Unauthorized Constr.", key="layer_unauthorized")
        st.toggle("Encroachment", key="layer_encroachment")
        st.toggle("Dispute Claims", key="layer_disputes")
    with l2:
        st.toggle("LiDAR Confidence", key="layer_confidence")
        st.toggle("Underground Infra", key="layer_underground")
        st.toggle("Change Detection", key="layer_change_detection")

    st.markdown("---")
    st.markdown("### BASEMENT / UNDERGROUND FLOORS")
    st.toggle(
        "Show Underground Floors", key="show_underground_floors",
        help="Renders the SELECTED building's real underground/basement levels "
             "(set the count in the Structure Editor workspace) as stratified "
             "slabs below grade — separate from the generic 'Underground Infra' "
             "utility-line layer above."
    )
    if st.session_state.show_underground_floors:
        st.caption(
            "⚠️ The globe is a solid sphere, not real terrain, so an opaque "
            "GROUND plane will visually cover basement levels from above. "
            "GROUND is auto-hidden while a building is open here so the "
            "underground slabs stay visible; toggle it back on for the city view."
        )

    st.multiselect(
        "Urban Asset Layers", ASSET_LAYER_OPTIONS, key="active_asset_layers"
    )

    st.selectbox(
        "Disaster Simulation Overlay", ["None"] + DISASTER_TYPES, key="disaster_mode"
    )

    tm1, tm2 = st.columns([1, 2])
    with tm1:
        st.toggle("Time Machine", key="time_machine_active")
    with tm2:
        if st.session_state.time_machine_active:
            st.slider(
                "Year", min_value=2005, max_value=2026, key="time_machine_year", label_visibility="collapsed"
            )

    st.markdown("---")
    st.markdown("### SURVEY INGEST")
    uploaded_file = st.file_uploader(
        "LAS / LAZ point cloud",
        type=["las", "laz"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        st.caption(f"Queued: {uploaded_file.name}")
        if st.button("IMPORT SURVEY", use_container_width=True):
            with st.spinner("Extracting ground / vegetation / building classes..."):
                try:
                    bytes_data = uploaded_file.read()
                    parsed = process_uploaded_las(bytes_data, uploaded_file.name)
                    st.session_state.city_model = parsed
                    reset_selection()
                    st.success(
                        f"{len(parsed.buildings)} buildings • "
                        f"{len(parsed.ground_elements)} ground cells • "
                        f"{len(parsed.vegetation_elements)} vegetation clusters"
                    )
                    st.rerun()
                except Exception as err:
                    st.error(f"Survey import failed: {err}")

    if st.button("LOAD BENCHMARK CITY", use_container_width=True):
        st.session_state.city_model = generate_synthetic_city()
        reset_selection()
        st.rerun()

    st.markdown("---")
    st.caption(
        "💡 2D→3D floor plan and AutoCAD (DXF) wall import live in the "
        "**'2D→3D / CAD Import'** workspace above — pick a building/floor there."
    )

    st.markdown("---")
    st.markdown("### RECORD LOOKUP")
    st.text_input(
        "ULPIN / Unit / Building",
        key="search_query",
        placeholder="e.g. ULPIN-KA-0001...",
    )
    if st.session_state.search_query.strip():
        q = st.session_state.search_query.strip().lower()
        matches = []
        for b, f, u in _all_units(st.session_state.city_model):
            hay = " ".join([u.ulpin, u.unit_id, b.building_id, f.floor_id, u.owner_name]).lower()
            if q in hay:
                matches.append((b, f, u))
        st.caption(f"{len(matches)} record(s) found")
        if matches:
            labels = [f"{u.ulpin}  |  {b.building_id}  |  F{f.floor_number}" for b, f, u in matches[:30]]
            pick = st.selectbox("Matching records", labels)
            idx = labels.index(pick)
            b, f, u = matches[idx]
            if st.button("OPEN CADASTRAL RECORD", use_container_width=True):
                st.session_state.selected_building_id = b.building_id
                st.session_state.selected_floor_id = f.floor_id
                st.session_state.selected_unit_id = u.unit_id
                st.session_state.current_level = "UNIT"
                st.rerun()

    st.markdown("---")
    st.caption(
        "Prototype note: compliance indicators and ownership records shown here are "
        "synthetic/demo records unless populated from an authoritative land registry."
    )

b_id = st.session_state.selected_building_id
f_id = st.session_state.selected_floor_id
u_id = st.session_state.selected_unit_id

# ------------------------------------------------------------------
# Section 8 — derive analytical layers for the CURRENT city model.
# Purely additive: computed alongside the untouched pipeline outputs.
# ------------------------------------------------------------------
_city = st.session_state.city_model
status_map = compute_unit_status_map(_city, _city.city_id)
building_intel = compute_building_intel(_city, _city.city_id)
change_events = simulate_change_detection(_city, _city.city_id)

_overlay_boxes: List[dict] = []
_overlay_points: List[dict] = []

if st.session_state.layer_unauthorized:
    _overlay_boxes += build_unauthorized_overlays(_city, building_intel)
if st.session_state.layer_encroachment:
    _overlay_boxes += build_encroachment_overlays(_city, building_intel)
if st.session_state.layer_confidence:
    _overlay_points += build_confidence_heatmap_points(_city, building_intel)
if st.session_state.layer_underground:
    _overlay_boxes += build_underground_overlays(_city)
if st.session_state.layer_disputes:
    _overlay_boxes += build_dispute_overlays(_city, building_intel)
if st.session_state.layer_change_detection:
    _overlay_boxes += build_change_detection_overlays(_city, change_events)
if st.session_state.active_asset_layers:
    _overlay_boxes += build_urban_asset_layer_overlays(_city, st.session_state.active_asset_layers)
if st.session_state.disaster_mode != "None":
    _overlay_boxes += build_disaster_overlays(_city, building_intel, st.session_state.disaster_mode)
if st.session_state.time_machine_active and b_id and b_id in _city.buildings:
    tm_overlay, tm_state = build_time_machine_overlay(_city.buildings[b_id], _city, st.session_state.time_machine_year)
    _overlay_boxes += tm_overlay

# --- Level-aware overlays: underground floors and imported CAD walls belong
# to ONE elevation/floor each, so they must only appear on the viewport that
# actually corresponds to them. Without this, drilling into a single floor
# would still show the basement (wrong elevation) and every imported floor
# plan on the building (wrong floor) stacked on top of the current one.
_current_level = st.session_state.current_level

if (
    st.session_state.show_underground_floors
    and b_id and b_id in _city.buildings
    and _current_level == "BUILDING"
):
    _overlay_boxes += build_selected_building_underground_slabs(_city.buildings[b_id], _city)

if b_id in st.session_state.cad_import_overlays:
    _cad_for_bld = st.session_state.cad_import_overlays[b_id]
    if _current_level == "BUILDING":
        # Building-level view shows every floor at its real elevation, so
        # all imported floor plans for this building can coexist here.
        for _flr_overlays in _cad_for_bld.values():
            _overlay_boxes += _flr_overlays
    elif _current_level in ("FLOOR", "UNIT") and f_id in _cad_for_bld:
        # Drilled into one floor: only that floor's imported walls belong.
        _overlay_boxes += _cad_for_bld[f_id]

# GROUND auto-hide while inspecting underground floors on a selected
# building — see the sidebar note next to "Show Underground Floors".
_effective_show_ground = st.session_state.show_ground and not (
    st.session_state.show_underground_floors and b_id and b_id in _city.buildings
    and _current_level == "BUILDING"
)

crumb = f"🗺️ **CITY: {st.session_state.city_model.city_id}**"
if b_id:
    crumb += f" ❯ 🏢 **{b_id}**"
if f_id:
    crumb += f" ❯ 🥞 **{f_id.split('-')[-1]}**"
if u_id:
    crumb += f" ❯ 🔑 **UNIT {u_id.split('-')[-1]}**"

st.markdown(f"<div class='breadcrumb-bar'>{crumb}</div>", unsafe_allow_html=True)

nav_cols = st.columns([1, 1, 1, 3])
if b_id and nav_cols[0].button("⬅ Back to City"):
    st.session_state.current_level = "CITY"
    st.session_state.selected_building_id = None
    st.session_state.selected_floor_id = None
    st.session_state.selected_unit_id = None
    st.rerun()

if f_id and nav_cols[1].button("⬅ Back to Building"):
    st.session_state.current_level = "BUILDING"
    st.session_state.selected_floor_id = None
    st.session_state.selected_unit_id = None
    st.rerun()

if u_id and nav_cols[2].button("⬅ Back to Floor"):
    st.session_state.current_level = "FLOOR"
    st.session_state.selected_unit_id = None
    st.rerun()


# ------------------------------------------------------------------
# SIH26011 operational status strip
# ------------------------------------------------------------------
stats = cadastral_statistics(st.session_state.city_model)
status_cols = st.columns(6)
status_cols[0].metric("BUILDINGS", stats["buildings"])
status_cols[1].metric("FLOORS", stats["floors"])
status_cols[2].metric("3D UNITS", stats["units"])
status_cols[3].metric("FLOOR AREA", f'{stats["floor_area"]:.0f} m²')
status_cols[4].metric("AVG HEIGHT", f'{stats["avg_height"]:.1f} m')
status_cols[5].metric("DATUM", f'{st.session_state.city_model.ground_datum:.2f} m')

if st.session_state.workspace_mode == "Register":
    st.markdown("### PROPERTY REGISTER")
    reg = build_cadastral_register(st.session_state.city_model)
    rc1, rc2 = st.columns([5, 1])
    with rc1:
        st.dataframe(reg, use_container_width=True, height=430, hide_index=True)
    with rc2:
        st.write("**Record set**")
        st.write(f"{len(reg)} 3D spatial units")
        st.download_button(
            "EXPORT REGISTER CSV",
            reg.to_csv(index=False).encode("utf-8"),
            file_name="sih26011_cadastral_register.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.markdown("---")

elif st.session_state.workspace_mode == "Compliance":
    st.markdown("### DEVELOPMENT / COMPLIANCE SCREENING")
    st.caption("Prototype screening layer — not a statutory approval or legal determination.")
    compliance_rows = []
    for b in st.session_state.city_model.buildings.values():
        far, coverage, flags = building_compliance_snapshot(b)
        compliance_rows.append({
            "Building": b.building_id,
            "Height (m)": round(b.height, 1),
            "Floors": b.num_floors,
            "FAR": round(far, 2),
            "Coverage %": round(coverage, 1),
            "Construction": "Under construction" if b.is_under_construction else "Existing",
            "Review flags": ", ".join(flags) if flags else "—",
        })
    st.dataframe(pd.DataFrame(compliance_rows), use_container_width=True, height=360, hide_index=True)
    st.info("Use this layer to demonstrate how a 3D cadastral platform can support planning review, FAR/coverage checks and field inspection workflows.")

elif st.session_state.workspace_mode == "Provenance":
    st.markdown("### SURVEY PROVENANCE & DATA QUALITY")
    p1, p2, p3 = st.columns(3)
    p1.metric("SOURCE", "LAS / LAZ" if "SURVEY" in st.session_state.city_model.city_id else "BENCHMARK")
    p2.metric("GROUND CELLS", len(st.session_state.city_model.ground_elements))
    p3.metric("VEGETATION", len(st.session_state.city_model.vegetation_elements))
    st.write({
        "City / dataset": st.session_state.city_model.name,
        "City ID": st.session_state.city_model.city_id,
        "Reference longitude": st.session_state.city_model.ref_lon,
        "Reference latitude": st.session_state.city_model.ref_lat,
        "Ground datum": st.session_state.city_model.ground_datum,
        "Building extraction": "Class 6 clustering / benchmark cadastral envelopes",
        "Floor inference": "Height ÷ estimated floor height",
        "Cadastral model": "3D spatial units / ULPIN-style prototype identifiers",
        "Status": "Demonstration dataset — validate against authoritative survey and registry sources before operational use.",
    })
    st.markdown("---")

elif st.session_state.workspace_mode == "Ownership 3D":
    st.markdown("### 3D PROPERTY OWNERSHIP VISUALIZATION")
    st.caption("Units are colored by ownership status. Set Representation = OWNERSHIP_STATUS in the sidebar to see it in the 3D viewport below, or explode floors for the vertical property stack.")
    legend_cols = st.columns(len(UNIT_STATUSES))
    for col, status in zip(legend_cols, UNIT_STATUSES):
        col.markdown(
            f"<div style='background:{STATUS_COLORS[status]};color:#0B1018;border-radius:4px;"
            f"padding:6px;text-align:center;font-weight:700;font-size:12px'>{status}</div>",
            unsafe_allow_html=True,
        )
    counts = {s: sum(1 for v in status_map.values() if v == s) for s in UNIT_STATUSES}
    st.dataframe(pd.DataFrame([{"Status": s, "Units": counts[s]} for s in UNIT_STATUSES]),
                 use_container_width=True, hide_index=True)
    st.info("💡 Tip: enable 'CUTAWAY SPATIAL UNITS' + 'EXPLODE FLOOR STRATA' in the sidebar, then drill into a building to see the full City → Building → Floor → Unit vertical property stack with per-unit selection.")
    st.markdown("---")

elif st.session_state.workspace_mode == "Violations":
    st.markdown("### UNAUTHORIZED CONSTRUCTION + 3D ENCROACHMENT DETECTION")
    st.caption("Comparing approved plan height/footprint vs LiDAR-derived reconstruction, and building footprints vs recorded parcel boundaries.")
    viol_rows = []
    for b_id_v, bld_v in _city.buildings.items():
        v = building_intel[b_id_v]
        if v["is_unauthorized"] or v["is_encroaching"]:
            viol_rows.append({
                "Building": b_id_v,
                "Unauthorized Construction": "⚠️ Yes" if v["is_unauthorized"] else "—",
                "Excess Height (m)": round(v["excess_height"], 1),
                "Excess Footprint (m²)": round(v["excess_footprint"], 1),
                "Encroachment": "⚠️ Yes" if v["is_encroaching"] else "—",
                "Encroachment Area (m²)": round(v["encroachment_area"], 1),
            })
    if viol_rows:
        st.dataframe(pd.DataFrame(viol_rows), use_container_width=True, height=360, hide_index=True)
    else:
        st.success("No violations detected in the current dataset.")
    st.info("💡 Enable 'Unauthorized Constr.' and 'Encroachment' under 3D ANALYTICAL LAYERS in the sidebar to see these highlighted directly in the 3D viewport (orange = excess construction, red = boundary encroachment).")
    st.markdown("---")

elif st.session_state.workspace_mode == "Intelligence Dashboard":
    st.markdown("### URBAN INTELLIGENCE DASHBOARD")
    summary = urban_intelligence_summary(_city, status_map, building_intel)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("BUILDINGS", summary["buildings"])
    d2.metric("PROPERTY UNITS", summary["units"])
    d3.metric("AVG FLOORS", f'{summary["avg_floors"]:.1f}')
    d4.metric("AVG FAR", f'{summary["avg_far"]:.2f}')
    d5, d6, d7, d8 = st.columns(4)
    d5.metric("BUILT FOOTPRINT", f'{summary["built_footprint"]:.0f} m²')
    d6.metric("OPEN SPACE", f'{summary["open_space_pct"]:.1f} %')
    d7.metric("VACANT UNITS", summary["vacant_units"])
    d8.metric("VIOLATIONS", summary["unauthorized_buildings"] + summary["encroaching_buildings"])
    st.markdown("#### Ownership Status Distribution")
    status_df = pd.DataFrame([{"Status": s, "Units": sum(1 for v in status_map.values() if v == s)} for s in UNIT_STATUSES])
    st.bar_chart(status_df.set_index("Status"))
    st.markdown("#### Land Use / Risk Snapshot")
    r1, r2, r3 = st.columns(3)
    r1.metric("DISPUTED BUILDINGS", summary["disputed_buildings"])
    r2.metric("AVG LIDAR CONFIDENCE", f'{summary["avg_confidence"]:.2f}')
    r3.metric("COMMERCIAL UNITS", summary["commercial_units"])
    st.markdown("---")

elif st.session_state.workspace_mode == "FAR / Utilization":
    st.markdown("### FAR / LAND UTILIZATION ANALYSIS")
    st.caption("Floor Area Ratio, footprint, open space and density derived per building.")
    far_df = far_utilization_table(_city, building_intel)
    st.dataframe(far_df, use_container_width=True, height=400, hide_index=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("**FAR by building**")
        st.bar_chart(far_df.set_index("Building")["FAR"])
    with fc2:
        st.markdown("**Height distribution**")
        st.bar_chart(far_df.set_index("Building")["Height (m)"])
    st.download_button(
        "EXPORT FAR ANALYSIS CSV", far_df.to_csv(index=False).encode("utf-8"),
        file_name="sih26011_far_analysis.csv", mime="text/csv",
    )
    st.markdown("---")

elif st.session_state.workspace_mode == "Cadastral Copilot":
    st.markdown("### CADASTRAL COPILOT")
    st.caption("Ask natural-language questions about the current 3D cadastral dataset.")
    example_cols = st.columns(3)
    examples = ["Show vacant properties", "Which properties have violations?", "How many commercial units are here?"]
    for col, ex in zip(example_cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state["_copilot_query"] = ex
    copilot_query = st.text_input("Your question", value=st.session_state.get("_copilot_query", ""),
                                   placeholder="e.g. Which properties have violations?")
    if copilot_query:
        st.session_state["_copilot_query"] = copilot_query
        answer = cadastral_copilot_answer(copilot_query, _city, status_map)
        st.markdown(f"<div class='metric-card'>{answer}</div>", unsafe_allow_html=True)
    st.markdown("---")

elif st.session_state.workspace_mode == "Spatial Search":
    st.markdown("### 3D SPATIAL SEARCH")
    st.caption("Search by ULPIN, unit / building / owner ID, usage, floor, area, and more.")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        s_query = st.text_input("ULPIN / ID / Owner", key="ss_query")
    with sc2:
        s_usage = st.selectbox("Usage", ["Any", "Residential", "Commercial", "Common"], key="ss_usage")
    with sc3:
        s_status = st.selectbox("Status", ["Any"] + UNIT_STATUSES, key="ss_status")
    with sc4:
        s_floor = st.number_input("Floor (0 = any)", min_value=0, max_value=60, value=0, key="ss_floor")
    sc5, sc6 = st.columns(2)
    with sc5:
        s_min_area = st.number_input("Min area (m²)", min_value=0.0, value=0.0, key="ss_min_area")
    with sc6:
        s_max_area = st.number_input("Max area (m²)", min_value=0.0, value=1000.0, key="ss_max_area")

    results = spatial_search(
        _city, status_map, query=s_query, usage=s_usage, status_filter=s_status,
        min_area=s_min_area, max_area=s_max_area, floor=(s_floor if s_floor > 0 else None),
    )
    st.caption(f"{len(results)} matching record(s)")
    if results:
        res_df = pd.DataFrame([{
            "ULPIN": u.ulpin, "Unit ID": u.unit_id, "Building": b.building_id, "Floor": f.floor_number,
            "Usage": u.property_type, "Owner": u.owner_name, "Area (m²)": round(u.area_sqm, 1), "Status": s,
        } for b, f, u, s in results[:200]])
        st.dataframe(res_df, use_container_width=True, height=380, hide_index=True)
        pick_labels = [f"{u.ulpin} | {b.building_id} | F{f.floor_number}" for b, f, u, s in results[:200]]
        pick = st.selectbox("Open a result in 3D", ["-- Choose --"] + pick_labels)
        if pick != "-- Choose --":
            idx = pick_labels.index(pick)
            b, f, u, s = results[idx]
            if st.button("OPEN IN 3D VIEWER"):
                st.session_state.selected_building_id = b.building_id
                st.session_state.selected_floor_id = f.floor_id
                st.session_state.selected_unit_id = u.unit_id
                st.session_state.current_level = "UNIT"
                st.rerun()
    st.markdown("---")

elif st.session_state.workspace_mode == "Digital Passport":
    st.markdown("### DIGITAL PROPERTY PASSPORT")
    st.caption("ULPIN, parcel, unit, area, floor, usage, building, status and verification, with QR code.")
    if not u_id:
        st.info("Select a property unit (drill down City → Building → Floor → Unit, or use Spatial Search) to generate its Digital Property Passport.")
    else:
        b_p = _city.buildings[b_id]
        f_p = next((f for f in b_p.floors if f.floor_id == f_id), b_p.floors[0])
        u_p = next((u for u in f_p.units if u.unit_id == u_id), None)
        if u_p:
            status_p = status_map.get(u_p.unit_id, "Verified")
            passport = generate_property_passport(b_p, f_p, u_p, status_p)
            pc1, pc2 = st.columns([2, 1])
            with pc1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                for k, v in passport.items():
                    st.write(f"**{k}:** {v}")
                st.markdown("</div>", unsafe_allow_html=True)
            with pc2:
                qr_payload = " | ".join(f"{k}={v}" for k, v in passport.items())
                qr_b64 = make_qr_code_png_b64(qr_payload)
                if qr_b64:
                    st.image(f"data:image/png;base64,{qr_b64}", caption="Scan for property record", width=180)
                else:
                    st.warning("QR generation requires the `qrcode` package (pip install qrcode[pil]).")
                st.download_button(
                    "EXPORT PASSPORT JSON",
                    pd.Series(passport).to_json().encode("utf-8"),
                    file_name=f"{u_p.unit_id}_passport.json", mime="application/json",
                )
    st.markdown("---")

elif st.session_state.workspace_mode == "Time Machine":
    st.markdown("### PROPERTY TIME MACHINE")
    st.caption("Simulated historical construction states. Turn on 'Time Machine' in the sidebar and pick a building to see it animate in 3D.")
    if not b_id:
        st.info("Select a building to view its simulated timeline.")
    else:
        bld_tm = _city.buildings[b_id]
        states = property_time_machine_states(bld_tm)
        tl_cols = st.columns(len(states))
        for col, s in zip(tl_cols, states):
            active = s["year"] <= st.session_state.time_machine_year
            col.markdown(
                f"<div style='background:{s['color'] if active else '#1E293B'};color:#F8FAFC;border-radius:4px;"
                f"padding:8px;text-align:center;font-size:11px'><b>{s['year']}</b><br>{s['label']}</div>",
                unsafe_allow_html=True,
            )
        st.caption("Turn on 'Time Machine' + set the Year slider in the sidebar to render this state in the 3D viewport.")
    st.markdown("---")

elif st.session_state.workspace_mode == "Disaster Simulation":
    st.markdown("### DISASTER SIMULATION")
    st.caption("Flood, fire and earthquake risk simulation with affected-building highlighting. Pick a hazard in the sidebar's 'Disaster Simulation Overlay' selector.")
    if st.session_state.disaster_mode == "None":
        st.info("Select Flood, Fire or Earthquake from the sidebar to simulate affected buildings.")
    else:
        key = {"Flood": "flood_risk", "Fire": "fire_risk", "Earthquake": "eq_risk"}[st.session_state.disaster_mode]
        rows = [{"Building": b, "Risk Score": round(v[key], 2), "Affected": "Yes" if v[key] > 0.65 else "Monitor"}
                for b, v in building_intel.items() if v[key] > 0.5]
        rows.sort(key=lambda r: -r["Risk Score"])
        st.metric(f"BUILDINGS AT RISK ({st.session_state.disaster_mode.upper()})", len(rows))
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=340, hide_index=True)
    st.markdown("---")

elif st.session_state.workspace_mode == "Change Detection":
    st.markdown("### CONSTRUCTION CHANGE DETECTION")
    st.caption("Simulated comparison between two scan epochs, highlighting structural changes.")
    if change_events:
        chg_df = pd.DataFrame(change_events).rename(columns={
            "building_id": "Building", "delta_height": "Δ Height (m)", "change_type": "Change Type",
            "scan_a_date": "Scan A", "scan_b_date": "Scan B",
        })
        st.dataframe(chg_df, use_container_width=True, height=340, hide_index=True)
    else:
        st.success("No structural changes detected between the two simulated scan epochs.")
    st.info("💡 Enable 'Change Detection' under 3D ANALYTICAL LAYERS to see the deltas rendered on top of the affected buildings in 3D.")
    st.markdown("---")

elif st.session_state.workspace_mode == "Dispute Mode":
    st.markdown("### CADASTRAL DISPUTE MODE")
    st.caption("Buildings with competing parcel boundary claims and their review status.")
    disputed = [b for b, v in building_intel.items() if v["is_disputed"]]
    st.metric("DISPUTED BUILDINGS", len(disputed))
    if disputed:
        st.dataframe(pd.DataFrame([{
            "Building": b, "Claim A": "Registered Owner", "Claim B": "Competing Claimant",
            "Review Status": "Under Review",
        } for b in disputed]), use_container_width=True, height=320, hide_index=True)
    st.info("💡 Enable 'Dispute Claims' under 3D ANALYTICAL LAYERS to see the overlapping parcel claims rendered in 3D (indigo vs pink volumes).")
    st.markdown("---")

elif st.session_state.workspace_mode == "2D→3D / CAD Import":
    st.markdown("### 2D → 3D FLOOR PLAN & AUTOCAD (DXF) MAP IMPORT")
    st.caption(
        "Prototype-grade conversion: raster floor plans are converted with edge/contour "
        "detection (not full CAD vectorization), and DXF import reads LINE / LWPOLYLINE "
        "wall centerlines only. Both are heuristics meant for a quick massing check on "
        "top of the existing building — not survey-grade CAD reconstruction."
    )
    if not b_id:
        st.info("Select a building (drill into the 3D city view, or use Direct Building Selection) to import a floor plan onto it.")
    else:
        bld_ci = _city.buildings[b_id]
        flr_options_ci = [f.floor_id for f in bld_ci.floors]
        target_flr_id = st.selectbox("Target floor", flr_options_ci,
                                      index=(flr_options_ci.index(f_id) if f_id in flr_options_ci else 0))
        target_flr = next(f for f in bld_ci.floors if f.floor_id == target_flr_id)

        ci1, ci2 = st.columns(2)
        with ci1:
            st.markdown("**Raster floor plan (PNG/JPG)**")
            img_file = st.file_uploader("2D floor plan image", type=["png", "jpg", "jpeg"], key="ci_img_upload")
            wall_h_img = st.number_input("Wall height (m)", min_value=0.5, max_value=6.0, value=2.6, step=0.1, key="ci_wall_h_img")
            if img_file is not None and st.button("CONVERT IMAGE → 3D WALLS", use_container_width=True):
                try:
                    overlays, n = convert_floorplan_image_to_wall_overlays(
                        img_file.read(), bld_ci, target_flr, _city, wall_height=wall_h_img
                    )
                    st.session_state.cad_import_overlays.setdefault(b_id, {})[target_flr_id] = overlays
                    st.success(f"Detected {n} wall segment(s) and placed them on {target_flr_id}.")
                    st.rerun()
                except Exception as err:
                    st.error(f"Conversion failed: {err}")
        with ci2:
            st.markdown("**AutoCAD map (DXF wall drawing)**")
            dxf_file = st.file_uploader("DXF file", type=["dxf"], key="ci_dxf_upload")
            wall_h_dxf = st.number_input("Wall height (m)", min_value=0.5, max_value=6.0, value=2.6, step=0.1, key="ci_wall_h_dxf")
            if dxf_file is not None and st.button("CONVERT DXF → 3D WALLS", use_container_width=True):
                try:
                    dxf_text = dxf_file.read().decode("utf-8", errors="ignore")
                    overlays, n = parse_dxf_walls_to_overlays(
                        dxf_text, bld_ci, target_flr, _city, wall_height=wall_h_dxf
                    )
                    if n == 0:
                        st.warning("No LINE / LWPOLYLINE wall geometry found in that DXF.")
                    else:
                        st.session_state.cad_import_overlays.setdefault(b_id, {})[target_flr_id] = overlays
                        st.success(f"Read {n} wall segment(s) from DXF and placed them on {target_flr_id}.")
                        st.rerun()
                except Exception as err:
                    st.error(f"DXF parse failed: {err}")

        existing = st.session_state.cad_import_overlays.get(b_id, {})
        if existing:
            st.markdown("---")
            st.write(f"**Imported on this building:** {', '.join(f'{fid} ({len(v)} walls)' for fid, v in existing.items())}")
            if st.button("CLEAR IMPORTED FLOOR PLAN(S) FOR THIS BUILDING"):
                st.session_state.cad_import_overlays.pop(b_id, None)
                st.rerun()
        st.caption("Imported walls render as overlay boxes on the Floor / Unit 3D viewport below (drill into this building's floor to see them).")
    st.markdown("---")

elif st.session_state.workspace_mode == "Structure Editor":
    st.markdown("### FRONTEND STRUCTURAL & DATABASE EDITOR")
    st.caption("Directly edit building, floor and unit / cadastral records. Changes apply immediately to the in-memory city model.")
    if not b_id:
        st.info("Select a building to edit its structure and records.")
    else:
        bld_e = _city.buildings[b_id]
        st.markdown("#### Building")
        with st.form("edit_building_form"):
            e1, e2, e3 = st.columns(3)
            with e1:
                new_typology = st.selectbox("Typology / facade", list(INDIAN_PALETTES.keys()),
                                             index=list(INDIAN_PALETTES.keys()).index(bld_e.typology)
                                             if bld_e.typology in INDIAN_PALETTES else 0)
                new_floors = st.number_input("Floors (above ground)", min_value=1, max_value=80, value=int(bld_e.num_floors))
            with e2:
                new_underground = st.number_input("Underground / basement floors", min_value=0, max_value=6,
                                                    value=int(getattr(bld_e, "num_underground_floors", 0)))
                new_water_tank = st.color_picker("Water tank color", value=bld_e.water_tank_color
                                                  if bld_e.water_tank_color.startswith("#") else "#0F172A")
            with e3:
                new_awning = st.checkbox("Has commercial awning", value=bld_e.has_awning)
                new_balconies = st.checkbox("Has balconies", value=bld_e.has_balconies)
                new_under_construction = st.checkbox("Under construction", value=bld_e.is_under_construction)
            submitted_b = st.form_submit_button("SAVE BUILDING CHANGES", use_container_width=True)
            if submitted_b:
                if new_floors != bld_e.num_floors:
                    resize_building_floors(bld_e, new_floors)
                bld_e.num_underground_floors = int(new_underground)
                bld_e.typology = new_typology
                bld_e.color_palette = INDIAN_PALETTES[new_typology]
                bld_e.water_tank_color = new_water_tank
                bld_e.has_awning = new_awning
                bld_e.has_balconies = new_balconies
                bld_e.is_under_construction = new_under_construction
                st.success(f"{bld_e.building_id} updated.")
                st.rerun()

        st.markdown("#### Unit / Cadastral Record")
        flr_e_options = [f.floor_id for f in bld_e.floors]
        flr_e_pick = st.selectbox("Floor", flr_e_options,
                                   index=(flr_e_options.index(f_id) if f_id in flr_e_options else 0), key="se_floor_pick")
        flr_e = next(f for f in bld_e.floors if f.floor_id == flr_e_pick)
        unit_e_options = [u.unit_id for u in flr_e.units]
        if unit_e_options:
            unit_e_pick = st.selectbox("Unit", unit_e_options,
                                        index=(unit_e_options.index(u_id) if u_id in unit_e_options else 0), key="se_unit_pick")
            unit_e = next(u for u in flr_e.units if u.unit_id == unit_e_pick)
            with st.form("edit_unit_form"):
                u1, u2 = st.columns(2)
                with u1:
                    new_owner = st.text_input("Owner name", value=unit_e.owner_name)
                    new_ulpin = st.text_input("ULPIN", value=unit_e.ulpin)
                with u2:
                    new_rights = st.selectbox("Rights / tenure type",
                                               ["Freehold Title", "Commercial Lease", "Condominium Common"],
                                               index=["Freehold Title", "Commercial Lease", "Condominium Common"].index(unit_e.rights_type)
                                               if unit_e.rights_type in ["Freehold Title", "Commercial Lease", "Condominium Common"] else 0)
                    new_ptype = st.text_input("Property type / usage", value=unit_e.property_type)
                submitted_u = st.form_submit_button("SAVE UNIT CHANGES", use_container_width=True)
                if submitted_u:
                    unit_e.owner_name = new_owner
                    unit_e.ulpin = new_ulpin
                    unit_e.rights_type = new_rights
                    unit_e.property_type = new_ptype
                    st.success(f"{unit_e.unit_id} updated.")
                    st.rerun()
        else:
            st.info("This floor has no cadastral units.")
    st.markdown("---")

elif st.session_state.workspace_mode == "Analytics Dashboard":
    st.markdown("### ENHANCED DATA VISUALIZATION & ANALYTICS")
    st.caption("Additional charts on top of the Intelligence Dashboard, built from the live city model.")
    rows_an = []
    for b in _city.buildings.values():
        far, coverage, _flags = building_compliance_snapshot(b)
        rows_an.append({
            "Building": b.building_id, "Floors": b.num_floors,
            "Underground Floors": getattr(b, "num_underground_floors", 0),
            "Height (m)": b.height, "FAR": far, "Coverage %": coverage,
            "Footprint (m²)": b.width * b.depth,
            "Typology": b.typology.replace("_", " ").title(),
        })
    an_df = pd.DataFrame(rows_an)

    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("**Floors per building**")
        st.bar_chart(an_df.set_index("Building")["Floors"])
    with ac2:
        st.markdown("**Height vs. Footprint**")
        st.scatter_chart(an_df, x="Footprint (m²)", y="Height (m)", color="Typology")

    ac3, ac4 = st.columns(2)
    with ac3:
        st.markdown("**Buildings by typology**")
        st.bar_chart(an_df.groupby("Typology").size().rename("Buildings"))
    with ac4:
        st.markdown("**Buildings with underground floors**")
        ug_counts = an_df["Underground Floors"].value_counts().sort_index()
        ug_counts.index = [f"{i} level(s)" for i in ug_counts.index]
        st.bar_chart(ug_counts.rename("Buildings"))

    st.markdown("**FAR distribution (all buildings)**")
    st.area_chart(an_df.sort_values("FAR")[["FAR"]].reset_index(drop=True))

    unit_rows_an = [{
        "Property Type": u.property_type, "Area (m²)": u.area_sqm,
    } for _b, _f, u in _all_units(_city)]
    unit_df_an = pd.DataFrame(unit_rows_an)
    if not unit_df_an.empty:
        st.markdown("**Average unit area by property type**")
        st.bar_chart(unit_df_an.groupby("Property Type")["Area (m²)"].mean())
    st.markdown("---")

elif st.session_state.workspace_mode == "AutoCAD Export":
    st.markdown("### AUTOCAD EXPORT")
    st.caption(
        "Exports the selected building's floor + unit footprints as a DXF file "
        "(3D polylines, one layer per floor) — DXF opens directly in AutoCAD via "
        "File → Open, and is the standard interchange format for third-party tools "
        "since binary .dwg is a closed Autodesk format. Re-export from AutoCAD as "
        ".dwg after opening if a native file is required downstream."
    )
    if not b_id:
        st.info("Select a building to export.")
    else:
        bld_x = _city.buildings[b_id]
        xc1, xc2, xc3 = st.columns(3)
        xc1.metric("FLOORS", bld_x.num_floors)
        xc2.metric("UNITS", sum(len(f.units) for f in bld_x.floors))
        xc3.metric("FOOTPRINT", f"{bld_x.width * bld_x.depth:.0f} m²")
        dxf_bytes = export_building_to_dxf(bld_x)
        st.download_button(
            "EXPORT FOR AUTOCAD (.dxf)",
            dxf_bytes,
            file_name=f"{bld_x.building_id}_export.dxf",
            mime="application/dxf",
            use_container_width=True,
        )
    st.markdown("---")

view_col, info_col = st.columns([3, 1])

# ----------------------------------------------------------------------------
# 1. CITY LEVEL VIEWPORT
# ----------------------------------------------------------------------------
if st.session_state.current_level == "CITY":
    with view_col:
        st.subheader("Geospatial City Model (CesiumJS 3D)")
        st.caption("💡 Click any building in 3D to drill down.")
        clicked = render_cesium_viewer(
            st.session_state.city_model, "CITY",
            render_mode=st.session_state.render_mode,
            selected_bld_id=b_id,
            show_ground=_effective_show_ground,
            show_vegetation=st.session_state.show_vegetation,
            status_map=status_map,
            overlay_boxes=_overlay_boxes,
            overlay_points=_overlay_points,
        )

        if clicked and clicked.get("clicked_id"):
            clicked_id = clicked["clicked_id"]
        
            # Analytical overlays use IDs such as:
            # BLD-0047-UNAUTH
            # BLD-0047-ENCR
            # etc.
            # Convert them back to the real building ID before drilling down.
        
            buildings = st.session_state.city_model.buildings
        
            if clicked_id in buildings:
                # Normal building click
                building_id = clicked_id
        
            elif clicked_id.endswith("-UNAUTH"):
                # Unauthorized-construction overlay
                building_id = clicked_id[:-len("-UNAUTH")]
        
            elif clicked_id.endswith("-ENCR"):
                # Encroachment overlay
                building_id = clicked_id[:-len("-ENCR")]
        
            elif "-CHANGE-" in clicked_id:
                # Change-detection overlay, if applicable
                building_id = clicked_id.split("-CHANGE-")[0]
        
            else:
                # Unknown analytical object — don't attempt to open it as a building
                building_id = None
        
            if building_id and building_id in buildings:
                st.session_state.selected_building_id = building_id
                st.session_state.selected_floor_id = None
                st.session_state.selected_unit_id = None
                st.session_state.current_level = "BUILDING"
                st.rerun()

    with info_col:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Jurisdiction Details")
        st.write(f"**City:** {st.session_state.city_model.name}")
        st.write(f"**Active Mode:** `{st.session_state.render_mode}`")
        st.write(f"**Buildings (Class 6):** {len(st.session_state.city_model.buildings)}")
        st.write(f"**Ground Blocks (Class 2):** {len(st.session_state.city_model.ground_elements)}")
        st.write(f"**Vegetation (Class 3/5):** {len(st.session_state.city_model.vegetation_elements)}")
        total_p = sum(len(f.units) for b in st.session_state.city_model.buildings.values() for f in b.floors)
        st.write(f"**Registered 3D Parcels:** {total_p}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Direct Building Selection")
        b_choice = st.selectbox(
            "Inspect Building:",
            ["-- Choose --"] + list(st.session_state.city_model.buildings.keys()),
            index=0 if not b_id else (list(st.session_state.city_model.buildings.keys()).index(b_id) + 1)
        )
        if b_choice != "-- Choose --" and b_choice != b_id:
            st.session_state.selected_building_id = b_choice
            st.session_state.current_level = "BUILDING"
            st.rerun()

# ----------------------------------------------------------------------------
# 2. BUILDING LEVEL VIEWPORT
# ----------------------------------------------------------------------------
elif st.session_state.current_level == "BUILDING":
    bld_obj = st.session_state.city_model.buildings[st.session_state.selected_building_id]

    with view_col:
        st.subheader(f"Building Stratification: {bld_obj.building_id}")
        st.caption("💡 Click any floor slab or core in 3D to inspect internal spatial parcels.")
        clicked = render_cesium_viewer(
            st.session_state.city_model, "BUILDING",
            render_mode=st.session_state.render_mode,
            selected_bld_id=bld_obj.building_id,
            selected_flr_id=st.session_state.selected_floor_id,
            exploded=st.session_state.explode_floors,
            cutaway=st.session_state.cutaway_mode,
            show_ground=_effective_show_ground,
            show_vegetation=st.session_state.show_vegetation,
            status_map=status_map,
            overlay_boxes=_overlay_boxes,
            overlay_points=_overlay_points,
            imported_floor_ids=set(st.session_state.cad_import_overlays.get(bld_obj.building_id, {}).keys()),
        )

        if clicked and clicked.get("clicked_id"):
            clicked_id = clicked["clicked_id"]
            if clicked.get("level") == "FLOOR":
                st.session_state.selected_floor_id = clicked_id
                st.session_state.selected_unit_id = None
                st.session_state.current_level = "FLOOR"
                st.rerun()
            elif clicked.get("level") == "UNIT":
                st.session_state.selected_unit_id = clicked_id
                st.session_state.current_level = "UNIT"
                st.rerun()

    with info_col:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Building Specifications")
        st.write(f"**Building ID:** `{bld_obj.building_id}`")
        st.write(f"**Total Height:** {bld_obj.height:.2f} m")
        st.write(f"**Floors:** {bld_obj.num_floors}")
        st.write(f"**Typology:** {bld_obj.typology.replace('_', ' ').title()}")
        st.write(f"**Commercial Awning:** `{'Yes' if bld_obj.has_awning else 'No'}`")
        st.write(f"**Balconies:** `{'Yes' if bld_obj.has_balconies else 'No'}`")
        st.write(f"**Water Tank:** `{bld_obj.water_tank_color}`")
        st.write(f"**Under Construction:** `{'Yes' if bld_obj.is_under_construction else 'No'}`")
        st.markdown("</div>", unsafe_allow_html=True)

        far, coverage, flags = building_compliance_snapshot(bld_obj)
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Planning / Survey Indicators")
        st.write(f"**Estimated FAR:** {far:.2f}")
        st.write(f"**Estimated Ground Coverage:** {coverage:.1f}%")
        st.write(f"**Survey Review Flags:** {', '.join(flags) if flags else 'None'}")
        st.write("**Data Status:** Prototype / synthetic cadastral attributes")
        st.markdown("</div>", unsafe_allow_html=True)

        flr_options = [f.floor_id for f in bld_obj.floors]
        flr_pick = st.selectbox(
            "Select Stratified Floor:",
            ["-- Choose --"] + flr_options,
            index=0 if not f_id else (flr_options.index(f_id) + 1)
        )
        if flr_pick != "-- Choose --" and flr_pick != f_id:
            st.session_state.selected_floor_id = flr_pick
            st.session_state.current_level = "FLOOR"
            st.rerun()

# ----------------------------------------------------------------------------
# 3. FLOOR / UNIT LEVEL VIEWPORT
# ----------------------------------------------------------------------------
elif st.session_state.current_level in ["FLOOR", "UNIT"]:
    bld_obj = st.session_state.city_model.buildings[st.session_state.selected_building_id]
    flr_obj = next((f for f in bld_obj.floors if f.floor_id == st.session_state.selected_floor_id), bld_obj.floors[0])

    with view_col:
        st.subheader(f"Floor Cadastral Layout: {flr_obj.floor_id}")
        st.caption("💡 Click any unit in 3D to load its legal ULPIN title deed.")
        clicked = render_cesium_viewer(
            st.session_state.city_model, "FLOOR",
            render_mode=st.session_state.render_mode,
            selected_bld_id=bld_obj.building_id,
            selected_flr_id=flr_obj.floor_id,
            selected_unit_id=st.session_state.selected_unit_id,
            show_ground=_effective_show_ground,
            show_vegetation=st.session_state.show_vegetation,
            status_map=status_map,
            overlay_boxes=_overlay_boxes,
            overlay_points=_overlay_points,
            imported_floor_ids=set(st.session_state.cad_import_overlays.get(bld_obj.building_id, {}).keys()),
        )

        if clicked and clicked.get("clicked_id"):
            st.session_state.selected_unit_id = clicked["clicked_id"]
            st.session_state.current_level = "UNIT"
            st.rerun()

    with info_col:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Floor & Unit Register")
        st.write(f"**Floor Level:** Level {flr_obj.floor_number}")
        st.write(f"**Local Elevation:** {flr_obj.elevation:.1f} m")
        st.write(f"**Parcels on Level:** {len(flr_obj.units)}")
        st.markdown("</div>", unsafe_allow_html=True)

        unit_opts = [u.unit_id for u in flr_obj.units]
        u_pick = st.selectbox(
            "Select Legal 3D Unit:",
            ["-- Choose --"] + unit_opts,
            index=0 if not u_id else (unit_opts.index(u_id) + 1)
        )
        if u_pick != "-- Choose --" and u_pick != u_id:
            st.session_state.selected_unit_id = u_pick
            st.session_state.current_level = "UNIT"
            st.rerun()

        if st.session_state.selected_unit_id:
            active_u = next((u for u in flr_obj.units if u.unit_id == st.session_state.selected_unit_id), None)
            if active_u:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.write(f"**ULPIN:** `{active_u.ulpin}`")
                st.write(f"**Owner:** {active_u.owner_name}")
                st.write(f"**Tenure Class:** `{active_u.rights_type}`")
                st.write(f"**Parcel Type:** {active_u.property_type}")
                st.write(f"**Volume:** {active_u.volume_cum:.1f} m³")
                st.write(f"**Floor Area:** {active_u.area_sqm:.1f} m²")
                st.write(f"**Vertical Span:** {active_u.z_min:.1f}m to {active_u.z_max:.1f}m")
                st.write(f"**LADM Class:** `LA_SpatialUnit (3D)`")
                _u_status = status_map.get(active_u.unit_id, "Verified")
                st.markdown(
                    f"**Ownership Status:** "
                    f"<span style='background:{STATUS_COLORS.get(_u_status,'#94A3B8')};color:#0B1018;"
                    f"padding:2px 8px;border-radius:10px;font-weight:700;font-size:12px'>{_u_status}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                if st.button("📇 OPEN DIGITAL PASSPORT", use_container_width=True, key="open_passport_btn"):
                    st.session_state.workspace_mode = "Digital Passport"
                    st.rerun()
