"""
Real-Time ICU Deterioration Predictor & TreeSHAP Explainability Bridge
Consumes continuous HL7 vital streams, executes sub-millisecond XGBoost inference,
computes Native C++ TreeSHAP feature attributions, and formats clinical alerts.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import xgboost as xgb

# ---------------------------------------------------------------------------
# 1. Model Initialization
# ---------------------------------------------------------------------------
MODEL_FILE = "early_warning_15_30m.json"
if not os.path.exists(MODEL_FILE):
    alt_file = "early_warning_15_30m_3.json"
    if os.path.exists(alt_file):
        MODEL_FILE = alt_file
    else:
        raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found in current directory.")

# Load XGBoost booster once on module import
booster = xgb.Booster()
booster.load_model(MODEL_FILE)

# Exact 18 features expected by the trained model in exact order
MODEL_FEATURES: List[str] = [
    "HR",
    "O2Sat",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "FiO2",
    "pH",
    "PaCO2",
    "Lactate",
    "Shock_Index",
    "SF_Ratio",
    "ROX_Index",
    "delta_1h_HR",
    "delta_3h_HR",
    "delta_1h_MAP",
    "delta_3h_MAP",
    "delta_1h_Shock_Index",
]

# Clinical metadata and unit definitions
CLINICAL_META: Dict[str, Dict[str, str]] = {
    "HR": {"label": "Heart Rate", "unit": "bpm"},
    "O2Sat": {"label": "Oxygen Saturation (SpO2)", "unit": "%"},
    "SBP": {"label": "Systolic Blood Pressure", "unit": "mmHg"},
    "MAP": {"label": "Mean Arterial Pressure", "unit": "mmHg"},
    "DBP": {"label": "Diastolic Blood Pressure", "unit": "mmHg"},
    "Resp": {"label": "Respiratory Rate", "unit": "breaths/min"},
    "FiO2": {"label": "Fraction of Inspired O2", "unit": ""},
    "pH": {"label": "Arterial Blood pH", "unit": ""},
    "PaCO2": {"label": "Partial Pressure of CO2", "unit": "mmHg"},
    "Lactate": {"label": "Serum Lactate", "unit": "mmol/L"},
    "Shock_Index": {"label": "Cardiovascular Shock Index", "unit": ""},
    "SF_Ratio": {"label": "SpO2 / FiO2 Ratio", "unit": ""},
    "ROX_Index": {"label": "ROX Index", "unit": ""},
    "delta_1h_HR": {"label": "1-Hour HR Drift", "unit": "bpm"},
    "delta_3h_HR": {"label": "3-Hour HR Drift", "unit": "bpm"},
    "delta_1h_MAP": {"label": "1-Hour MAP Drift", "unit": "mmHg"},
    "delta_3h_MAP": {"label": "3-Hour MAP Drift", "unit": "mmHg"},
    "delta_1h_Shock_Index": {"label": "1-Hour Shock Index Drift", "unit": ""},
}

# In-memory rolling history buffer for delta calculations per patient
# patient_id -> list of {"timestamp": datetime, "HR": float, "MAP": float, "Shock_Index": float}
_PATIENT_HISTORY: Dict[str, List[Dict[str, Any]]] = {}

# Warm up JIT & C++ compiled Native TreeSHAP engine
_warmup_matrix = xgb.DMatrix(
    np.full((1, len(MODEL_FEATURES)), np.nan, dtype=np.float32),
    feature_names=MODEL_FEATURES,
)
_ = booster.predict(_warmup_matrix)
_ = booster.predict(_warmup_matrix, pred_contribs=True)


# ---------------------------------------------------------------------------
# 2. Feature Extraction & Dynamic Calculations
# ---------------------------------------------------------------------------
def _extract_vital_value(vitals_dict: Dict[str, Any], key: str) -> Optional[float]:
    """
    Extracts numerical value whether passed as a raw float/int or dict {'value': float}.
    """
    val = vitals_dict.get(key)
    if val is None:
        # Case-insensitive fallback
        for k, v in vitals_dict.items():
            if k.lower() == key.lower():
                val = v
                break

    if val is None:
        return None

    if isinstance(val, dict) and "value" in val:
        val = val["value"]

    try:
        fval = float(val)
        if np.isnan(fval):
            return None
        return fval
    except (ValueError, TypeError):
        return None


def _update_and_get_deltas(
    patient_id: str,
    now: datetime,
    hr: Optional[float],
    map_val: Optional[float],
    si: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Updates patient vital history and calculates lookback deltas (1-hour and 3-hour).
    Returns (delta_1h_HR, delta_3h_HR, delta_1h_MAP, delta_3h_MAP, delta_1h_Shock_Index).
    """
    if patient_id not in _PATIENT_HISTORY:
        _PATIENT_HISTORY[patient_id] = []

    history = _PATIENT_HISTORY[patient_id]

    # Append current entry
    current_entry = {
        "timestamp": now,
        "HR": hr,
        "MAP": map_val,
        "Shock_Index": si,
    }
    history.append(current_entry)

    # Prune records older than 4 hours
    cutoff = now - timedelta(hours=4)
    _PATIENT_HISTORY[patient_id] = [h for h in history if h["timestamp"] >= cutoff]
    history = _PATIENT_HISTORY[patient_id]

    def _find_lookback(metric: str, target_delta_minutes: int, tolerance_minutes: int = 20) -> Optional[float]:
        target_time = now - timedelta(minutes=target_delta_minutes)
        closest_diff = None
        closest_val = None
        for h in history[:-1]:  # Exclude current reading
            val = h.get(metric)
            if val is not None:
                diff_sec = abs((h["timestamp"] - target_time).total_seconds())
                if diff_sec <= tolerance_minutes * 60:
                    if closest_diff is None or diff_sec < closest_diff:
                        closest_diff = diff_sec
                        closest_val = val
        return closest_val

    # 1h lookbacks
    h1_hr = _find_lookback("HR", 60)
    h1_map = _find_lookback("MAP", 60)
    h1_si = _find_lookback("Shock_Index", 60)

    # 3h lookbacks
    h3_hr = _find_lookback("HR", 180)
    h3_map = _find_lookback("MAP", 180)

    d1_hr = (hr - h1_hr) if (hr is not None and h1_hr is not None) else None
    d3_hr = (hr - h3_hr) if (hr is not None and h3_hr is not None) else None
    d1_map = (map_val - h1_map) if (map_val is not None and h1_map is not None) else None
    d3_map = (map_val - h3_map) if (map_val is not None and h3_map is not None) else None
    d1_si = (si - h1_si) if (si is not None and h1_si is not None) else None

    return d1_hr, d3_hr, d1_map, d3_map, d1_si


# ---------------------------------------------------------------------------
# 3. Predict & Explain Engine (Sub-Millisecond Inference)
# ---------------------------------------------------------------------------
def predict_and_explain(patient_id: str, vitals: Dict[str, Any], timestamp: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Processes incoming continuous vitals, runs low-latency native TreeSHAP,
    and prints a formatted clinical telemetry alert block to the terminal.
    """
    t0 = time.perf_counter()
    now = timestamp or datetime.now()

    # 1. Extract Continuous Vitals
    hr = _extract_vital_value(vitals, "HR")
    o2sat = _extract_vital_value(vitals, "O2Sat")
    sbp = _extract_vital_value(vitals, "SBP")
    map_val = _extract_vital_value(vitals, "MAP")
    dbp = _extract_vital_value(vitals, "DBP")
    resp = _extract_vital_value(vitals, "Resp")

    # FiO2 defaults to 0.21 (room air) if not provided
    fio2 = _extract_vital_value(vitals, "FiO2") or 0.21

    # Intermittent labs default to np.nan so XGBoost follows default_left learned paths
    ph = _extract_vital_value(vitals, "pH")
    paco2 = _extract_vital_value(vitals, "PaCO2")
    lactate = _extract_vital_value(vitals, "Lactate")

    # 2. Dynamic Derived Clinical Indicators
    # Shock Index = HR / SBP
    shock_index = (hr / sbp) if (hr is not None and sbp is not None and sbp > 0) else None

    # S/F Ratio = SpO2 / FiO2
    sf_ratio = (o2sat / fio2) if (o2sat is not None and fio2 > 0) else None

    # ROX Index = (SpO2 / FiO2) / Resp
    rox_index = (sf_ratio / resp) if (sf_ratio is not None and resp is not None and resp > 0) else None

    # 3. Lookback Deltas
    d1_hr, d3_hr, d1_map, d3_map, d1_si = _update_and_get_deltas(patient_id, now, hr, map_val, shock_index)

    # 4. Feature Vector Assembly (matching strict 18-feature model schema)
    feature_map: Dict[str, Optional[float]] = {
        "HR": hr,
        "O2Sat": o2sat,
        "SBP": sbp,
        "MAP": map_val,
        "DBP": dbp,
        "Resp": resp,
        "FiO2": fio2,
        "pH": ph,
        "PaCO2": paco2,
        "Lactate": lactate,
        "Shock_Index": shock_index,
        "SF_Ratio": sf_ratio,
        "ROX_Index": rox_index,
        "delta_1h_HR": d1_hr,
        "delta_3h_HR": d3_hr,
        "delta_1h_MAP": d1_map,
        "delta_3h_MAP": d3_map,
        "delta_1h_Shock_Index": d1_si,
    }

    vector = [feature_map[f] if feature_map[f] is not None else np.nan for f in MODEL_FEATURES]
    data_array = np.array([vector], dtype=np.float32)
    dmatrix = xgb.DMatrix(data_array, feature_names=MODEL_FEATURES)

    # 5. Native C++ TreeSHAP Execution (< 2 ms)
    # Returns (19,) array: 18 feature Shapley attributions + 1 base bias
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]
    feature_shaps = contribs[: len(MODEL_FEATURES)]
    base_bias = float(contribs[-1])

    raw_margin = float(np.sum(contribs))
    # Logistic probability
    risk_probability = float(1.0 / (1.0 + np.exp(-raw_margin)))

    # 6. Severity Categorization
    if risk_probability >= 0.75:
        severity_tier = "[CRITICAL DETERIORATION ALERT]"
    elif risk_probability >= 0.40:
        severity_tier = "[ELEVATED RISK]"
    else:
        severity_tier = "[HEMODYNAMICALLY STABLE]"

    # 7. Map TreeSHAP Contributions
    attributions = []
    positive_drivers = []
    protective_factors = []

    for i, feat in enumerate(MODEL_FEATURES):
        shap_val = float(feature_shaps[i])
        raw_v = feature_map[feat]
        meta = CLINICAL_META.get(feat, {"label": feat, "unit": ""})
        entry = {
            "feature": feat,
            "label": meta["label"],
            "value": raw_v,
            "unit": meta["unit"],
            "shap": shap_val,
        }
        attributions.append(entry)
        if shap_val > 0.001:
            positive_drivers.append(entry)
        elif shap_val < -0.001:
            protective_factors.append(entry)

    # Sort positive drivers descending (highest risk first)
    positive_drivers.sort(key=lambda x: x["shap"], reverse=True)
    # Sort protective factors ascending (most protective first)
    protective_factors.sort(key=lambda x: x["shap"])

    top_pos = positive_drivers[:3]
    top_neg = protective_factors[:2]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # 8. Terminal Output Formatting
    _print_clinical_alert(
        patient_id=patient_id,
        timestamp=now,
        risk_prob=risk_probability,
        severity=severity_tier,
        feature_map=feature_map,
        top_pos=top_pos,
        top_neg=top_neg,
        elapsed_ms=elapsed_ms,
    )

    return {
        "patient_id": patient_id,
        "timestamp": now.isoformat(),
        "risk_probability": round(risk_probability, 4),
        "severity_tier": severity_tier,
        "top_risk_drivers": top_pos,
        "top_protective_factors": top_neg,
        "all_attributions": attributions,
        "base_bias": round(base_bias, 4),
        "raw_margin": round(raw_margin, 4),
        "latency_ms": round(elapsed_ms, 3),
    }


def _print_clinical_alert(
    patient_id: str,
    timestamp: datetime,
    risk_prob: float,
    severity: str,
    feature_map: Dict[str, Optional[float]],
    top_pos: List[Dict[str, Any]],
    top_neg: List[Dict[str, Any]],
    elapsed_ms: float,
):
    """
    Renders a clinical dashboard block to the console.
    """
    time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    hr_str = f"{feature_map['HR']:.0f} bpm" if feature_map["HR"] is not None else "N/A"
    sbp_str = f"{feature_map['SBP']:.0f}" if feature_map["SBP"] is not None else "N/A"
    dbp_str = f"{feature_map['DBP']:.0f}" if feature_map["DBP"] is not None else "N/A"
    map_str = f"{feature_map['MAP']:.0f} mmHg" if feature_map["MAP"] is not None else "N/A"
    o2_str = f"{feature_map['O2Sat']:.0f}%" if feature_map["O2Sat"] is not None else "N/A"
    rr_str = f"{feature_map['Resp']:.0f} bpm" if feature_map["Resp"] is not None else "N/A"
    si_str = f"{feature_map['Shock_Index']:.2f}" if feature_map["Shock_Index"] is not None else "N/A"
    rox_str = f"{feature_map['ROX_Index']:.1f}" if feature_map["ROX_Index"] is not None else "N/A"

    print("=" * 80)
    print(f"PATIENT MONITOR : {patient_id:<16} | Timestamp: {time_str} | Latency: {elapsed_ms:.2f} ms")
    print("-" * 80)
    print(f"Risk Assessment : {risk_prob:.1%} {severity}")
    print(f"Lead Time Window: 15 - 30 minutes pre-decompensation")
    print(f"Vitals Snapshot : HR: {hr_str} | BP: {sbp_str}/{dbp_str} (MAP: {map_str}) | SpO2: {o2_str} | RR: {rr_str}")
    print(f"Clinical Indices: Shock Index: {si_str} (norm < 0.90) | ROX Index: {rox_str}")

    print("\nTop Risk Drivers (Increasing Risk):")
    if top_pos:
        for idx, item in enumerate(top_pos, 1):
            val_str = f"{item['value']:.2f} {item['unit']}".strip() if item["value"] is not None else "N/A"
            print(f"  [+] {item['label']:<30}: {val_str:<14} (TreeSHAP: +{item['shap']:.3f})")
    else:
        print("  - None identified.")

    print("\nTop Protective Factors (Lowering Risk):")
    if top_neg:
        for idx, item in enumerate(top_neg, 1):
            val_str = f"{item['value']:.2f} {item['unit']}".strip() if item["value"] is not None else "N/A"
            print(f"  [-] {item['label']:<30}: {val_str:<14} (TreeSHAP: {item['shap']:.3f})")
    else:
        print("  - None identified.")

    print("=" * 80)


if __name__ == "__main__":
    print("\n--- Running Standalone Test on predictor.py ---")
    sample_decompensating = {
        "HR": {"value": 118, "units": "bpm"},
        "O2Sat": {"value": 92, "units": "%"},
        "SBP": {"value": 90, "units": "mmHg"},
        "MAP": {"value": 58, "units": "mmHg"},
        "DBP": {"value": 42, "units": "mmHg"},
        "Resp": {"value": 26, "units": "breaths/min"},
    }
    result = predict_and_explain("ICU-BED-04", sample_decompensating)
    print(f"Summary: Risk={result['risk_probability']:.1%}, Severity={result['severity_tier']}, Engine Latency={result['latency_ms']:.2f} ms")

