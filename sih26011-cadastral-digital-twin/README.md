# SIH26011 — 3D Cadastral & Land Administration Digital Twin

Prototype for **Smart India Hackathon 2026, Problem Statement SIH26011**.

A LADM (ISO 19152) aligned, procedurally generated 3D digital twin of an
Indian urban block — buildings, floors, and legally registered 3D spatial
units (parcels) — rendered with **CesiumJS** inside a **Streamlit** app.

## Features
- Procedural Indian urban digital twin (setbacks, balconies, chajjas,
  Sintex tanks, rebar, auto-rickshaws, window louvers, reinforced columns)
- 5 render/shader modes
- LADM `LA_SpatialUnit` (3D) style parcel registry with ULPIN, ownership,
  tenure class, and volumetric/area attributes
- Drill-down navigation: City → Building → Floor → Unit
- Unauthorized-construction and encroachment overlays, change detection
- Optional CAD floor-plan import (image-based wall detection via OpenCV)
- Optional LAS/LAZ point-cloud import
- Digital Passport view with QR code (optional)

## Project structure
```
.
├── app.py                  # Streamlit application
├── cesium_component/
│   └── index.html          # Custom Streamlit component: CesiumJS viewer
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Optional dependencies
- `laspy` — enables LAS/LAZ point-cloud import
- `qrcode` — enables QR code generation on the Digital Passport screen

The app checks for these at import time and disables the related features
gracefully if they aren't installed.

## Notes
- The CesiumJS viewer is loaded from Cesium's public CDN and does not
  require a Cesium ion access token for this prototype.
- All cadastral data (owners, ULPINs, statuses) is **synthetic**, generated
  for demonstration purposes only.

## License
Add a license of your choice (e.g. MIT) before publishing publicly.
