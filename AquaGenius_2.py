# app.py  –  AquaGenius WWTP Designer (refactored)
# ------------------------------------------------------------------
# 1.  pip install -r requirements.txt
# 2.  streamlit run app.py
# ------------------------------------------------------------------
import math
import io
from dataclasses import dataclass
from typing import Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import graphviz

# ------------------------- CONSTANTS ------------------------------
@dataclass(slots=True)
class C:
    # Conversions
    MGD_TO_M3D: float = 3785.41
    MLD_TO_M3D: float = 1000
    M3H_TO_GPM: float = 4.40287
    M3_TO_GAL:  float = 264.172
    M2_TO_FT2:  float = 10.7639

    # Kinetics
    Y:   float = 0.60
    KD:  float = 0.06
    TSS_VSS: float = 1.25
    VSS_TSS: float = 0.8

    # Aeration
    O2_BOD: float = 1.5
    O2_N:   float = 4.57
    SOTE:   float = 0.30
    O2_AIR: float = 0.232
    RHO_AIR: float = 1.225

    # Chemicals
    ALUM_P: float = 9.7
    METH_N: float = 2.86
    NAOH_H2S: float = 2.5
    NAOCL_H2S: float = 4.5
    H2SO4_NH3: float = 0.6

# ------------------------- DATA MODEL -----------------------------
@dataclass(slots=True)
class Influent:
    flow: float           # raw value from user
    unit: str             # "MGD", "MLD", "m³/day"
    bod:  float
    tss:  float
    tkn:  float
    tp:   float

    @property
    def m3d(self) -> float:
        return self.flow * {"MGD": C.MGD_TO_M3D,
                            "MLD": C.MLD_TO_M3D,
                            "m³/day": 1.0}[self.unit]

@dataclass(slots=True)
class Sizing:
    tech: str
    volume: float
    dims: Dict[str, Dict[str, str]]

# ----------------------- CALCULATIONS -----------------------------
@st.cache_data(show_spinner=False)
def calc_cas(inf: Influent) -> Tuple[Sizing, Dict]:
    srt, mlss, hrt = 10, 3500, 6
    vol = inf.m3d * hrt / 24
    clar_area = inf.m3d / 24
    dims = {
        "Anoxic":  _rect(vol*0.3),
        "Aerobic": _rect(vol*0.7),
        "Clarifier": _circ(clar_area, depth=4.5)
    }
    eff = {"BOD": 10, "TSS": 12, "TKN": 8, "TP": 2}
    res = {f"Effluent {k} (mg/L)": v for k, v in eff.items()}
    res["Required Air (m³/h)"] = _air_demand(inf, eff)
    return Sizing("CAS", vol, dims), res

@st.cache_data(show_spinner=False)
def calc_ifas(inf: Influent) -> Tuple[Sizing, Dict]:
    hrt = 6
    vol = inf.m3d * hrt / 24
    clar_area = inf.m3d / 28
    dims = {
        "Anoxic": _rect(vol*0.3),
        "IFAS":   _rect(vol*0.7),
        "Clarifier": _circ(clar_area, depth=4.5)
    }
    eff = {"BOD": 8, "TSS": 10, "TKN": 5, "TP": 1.5}
    res = {f"Effluent {k} (mg/L)": v for k, v in eff.items()}
    res["Required Air (m³/h)"] = _air_demand(inf, eff)
    return Sizing("IFAS", vol, dims), res

@st.cache_data(show_spinner=False)
def calc_mbr(inf: Influent) -> Tuple[Sizing, Dict]:
    hrt = 5
    vol = inf.m3d * hrt / 24
    dims = {
        "Anoxic": _rect(vol*0.4),
        "MBR":    _rect(vol*0.6),
    }
    eff = {"BOD": 5, "TSS": 1, "TKN": 4, "TP": 1}
    res = {f"Effluent {k} (mg/L)": v for k, v in eff.items()}
    res["Required Air (m³/h)"] = _air_demand(inf, eff)
    return Sizing("MBR", vol, dims), res

@st.cache_data(show_spinner=False)
def calc_mbbr(inf: Influent) -> Tuple[Sizing, Dict]:
    hrt = 4
    vol = inf.m3d * hrt / 24
    dims = {"MBBR": _rect(vol)}
    eff = {"BOD": 15, "TSS": 20, "TKN": 10, "TP": 2.5}
    res = {f"Effluent {k} (mg/L)": v for k, v in eff.items()}
    res["Required Air (m³/h)"] = _air_demand(inf, eff)
    return Sizing("MBBR", vol, dims), res

# ------------------ HELPERS --------------------------------------
def _rect(vol: float, depth: float = 4.5) -> Dict[str, str]:
    w = (vol / depth / 3) ** 0.5
    return {"Length (m)": f"{3*w:.1f}", "Width (m)": f"{w:.1f}", "Depth (m)": f"{depth:.1f}"}

def _circ(area: float, depth: float = 4.5) -> Dict[str, str]:
    d = 2 * math.sqrt(area / math.pi)
    return {"Diameter (m)": f"{d:.1f}", "SWD (m)": f"{depth:.1f}"}

def _air_demand(inf: Influent, eff: Dict[str, float]) -> float:
    bod_rm = inf.bod - eff["BOD"]
    n_rm   = inf.tkn - eff["TKN"]
    o2 = (bod_rm*C.O2_BOD + n_rm*C.O2_N) * inf.m3d / 1000
    return o2 / (C.SOTE * C.O2_AIR * C.RHO_AIR) / 24

# ------------------ PDF ------------------------------------------
def build_pdf(inf: Influent, sz: Sizing, res: Dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    story.append(Paragraph(f"{sz.tech} Design Summary", styles["Title"]))
    data = [[k, f"{v:.2f}"] for k, v in res.items()]
    tbl = Table([["Parameter", "Value"]] + data)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    story.append(tbl)
    doc.build(story)
    buffer.seek(0)
    return buffer.read()

# ------------------ UI -------------------------------------------
st.set_page_config("AquaGenius", "🌊", layout="wide")

with st.sidebar:
    st.title("⚙️ Influent Criteria")
    unit = st.selectbox("Unit", ["MGD", "MLD", "m³/day"])
    flow = st.number_input(f"Flow ({unit})", 0.1, 1e6, 1.0)
    bod  = st.number_input("BOD (mg/L)", 50, 1000, 250)
    tss  = st.number_input("TSS (mg/L)", 50, 1000, 220)
    tkn  = st.number_input("TKN (mg/L)", 10, 200, 40)
    tp   = st.number_input("TP (mg/L)",  1, 50, 7)
    inf  = Influent(flow, unit, bod, tss, tkn, tp)
    run  = st.button("Generate Design", use_container_width=True, type="primary")

if run:
    st.session_state["inf"] = inf

if "inf" in st.session_state:
    inf = st.session_state["inf"]
    tabs = st.tabs(["CAS", "IFAS", "MBR", "MBBR"])
    techs = [calc_cas, calc_ifas, calc_mbr, calc_mbbr]

    for tab, func in zip(tabs, techs):
        with tab:
            sz, res = func(inf)
            col1, col2 = st.columns(2)
            col1.metric("Volume", f"{sz.volume:,.0f} m³")
            col2.metric("Air", f"{res['Required Air (m³/h)']:,.0f} m³/h")
            with st.expander("Dimensions"):
                st.json(sz.dims)
            pdf = build_pdf(inf, sz, res)
            st.download_button("Download PDF", pdf,
                               file_name=f"{sz.tech}_report.pdf",
                               mime="application/pdf")
else:
    st.info("Configure influent criteria in the sidebar and click **Generate Design**.")