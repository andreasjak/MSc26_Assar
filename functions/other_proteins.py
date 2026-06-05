import joblib
from functions.analyte import ANALYTES
import pandas as pd
import numpy as np
from functions.analyte import get_ref

# ── Haptoglobin ─────────────────────────────────────────────────────────────

hapto_model  = joblib.load('../models/haptoglobin.pkl')
hapto_scaler = joblib.load('../models/haptoglobin_scaler.pkl')


def comment_haptoglobin(row: pd.DataFrame) -> str:
    haptoglobin = row['haptoglobin'].iloc[0]
    age         = row['age'].iloc[0]
    if haptoglobin < 0.24 and age <= 13:
        return "Sänkt haptoglobinhalt vilket kan ses vid såväl hemolys som leverpåverkan. "
    protein_cols  = [a.col for a in ANALYTES[:5]]
    protein_value = np.array(row[protein_cols].iloc[0]).reshape(1, -1)
    protein_value = hapto_scaler.transform(protein_value)
    prediction    = hapto_model.predict(protein_value)
    if prediction == 1:
        return "Konstellation av akutfasreaktanter förenlig med leverpåverkan och eller ökad erytrocytomsättning/hemolys. "
    return ""


# ── Kommentarsfunktioner ─────────────────────────────────────────────────────

def comment_albumin(row) -> str:
    albumin = row['albumin'][0]
    if albumin < 24: return "Grav hypoalbuminemi. "
    if albumin < 30: return "Påtaglig hypoalbuminemi. "
    if albumin < 34: return "Hypoalbuminemi. "
    return ""



def comment_immunglobulin(row: pd.DataFrame) -> str:
    comment = ""
    gender = row['gender'].iloc[0]

    igg_a = next(a for a in ANALYTES if a.col == 'igg')
    iga_a = next(a for a in ANALYTES if a.col == 'iga')
    igm_a = next(a for a in ANALYTES if a.col == 'igm')

    igg, iga, igm = row['igg'].iloc[0], row['iga'].iloc[0], row['igm'].iloc[0]

    igg_normal = get_ref(igg_a, gender)[0] <= igg <= get_ref(igg_a, gender)[1]
    iga_normal = get_ref(iga_a, gender)[0] <= iga <= get_ref(iga_a, gender)[1]
    igm_normal = get_ref(igm_a, gender)[0] <= igm <= get_ref(igm_a, gender)[1]

    if igg_normal and igm_normal and iga <= 0.07:
        return "IgA-halt <0.07 g/L inger misstanke om medfödd selektiv IgA-brist. "

    for analyte, value in [(igg_a, igg), (iga_a, iga), (igm_a, igm)]:
        low, high = get_ref(analyte, gender)
        if value < low:  comment += f"Sänkt halt av {analyte.name}. "
        if value > high: comment += f"Förhöjd halt av {analyte.name}. "

    if igg_normal and iga_normal and igm_normal:
        comment += "Immunglobuliner med normala halter. "

    return comment