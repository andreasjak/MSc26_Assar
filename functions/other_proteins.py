import joblib
from functions.analyte import ANALYTES
import pandas as pd
import numpy as np
from functions.analyte import get_ref
from functions.analyte import get_analyte

# ── Haptoglobin ─────────────────────────────────────────────────────────────

hapto_model  = joblib.load('../models/haptoglobin.pkl')
hapto_scaler = joblib.load('../models/haptoglobin_scaler.pkl')


def comment_haptoglobin(row: pd.DataFrame) -> str:
    haptoglobin = row['haptoglobin'].iloc[0]
    age         = row['age'].iloc[0]
    if haptoglobin < 0.24 and age > 13:
        return "Sänkt haptoglobinhalt vilket kan ses vid såväl hemolys som leverpåverkan. "
    
    antitrypsin_normal = row['antitrypsin'][0] / 1.5
    oros_normal = row['orosomukoid'][0] / 0.77
    hapto_normal = row['haptoglobin'][0] / 1.6

    anti_minus_oro = antitrypsin_normal - oros_normal
    anti_minus_hapto = antitrypsin_normal - hapto_normal
    oros_minus_hapto = oros_normal - hapto_normal


    leverprofil = ((anti_minus_oro > 0.05) & (anti_minus_hapto > 0.2) & (oros_minus_hapto > 0)).astype(int)
    if leverprofil == 1:
        return "Konstellation av akutfasreaktanter förenlig med leverpåverkan och eller ökad erytrocytomsättning/hemolys. "
    return ""

def comment_antitrypsin(row) -> str:
    antitrypsin = row['antitrypsin'][0]
    analyte = get_analyte('antitrypsin') 
    if antitrypsin < 0.7:
        return "Påtagligt sänkt halt av antitrypsin vilket inger misstanke om ärftlig antitrypsinbrist. Pi-typning bör övervägas. "
    if antitrypsin < get_ref(analyte,row['gender'][0])[0]:
        return "Lätt sänkt halt av antitrypsin. Önskar man påvisa/utesluta anlag för ärftlig antitrypsinbrist krävs Pi-typning. "
    return ""

# ── Kommentarsfunktioner ─────────────────────────────────────────────────────

def comment_albumin(row) -> str:
    albumin = row['albumin'][0]
    if albumin < 24: return "Grav hypoalbuminemi. "
    if albumin < 30: return "Påtaglig hypoalbuminemi. "
    if albumin < 34: return "Hypoalbuminemi. "
    return ""



# Gränser för lätt förhöjd/sänkt vs förhöjd/sänkt
ALARM_THRESHOLDS = {
    'igg': {'high': 17.5, 'low': 6.0},
    'iga': {'high': 6.0,  'low': 0.6},
    'igm': {'high': 3.0,  'low': 0.2},
}

def classify_ig(value, ref_low, ref_high, alarm_low, alarm_high):
    if value > alarm_high: return 'förhöjd'
    if value > ref_high:   return 'lätt förhöjd'
    if value < alarm_low:  return 'sänkt'
    if value < ref_low:    return 'lätt sänkt'
    return 'normal'

def comment_immunglobulin(row: pd.DataFrame) -> str:
    comment = ""
    gender = row['gender'].iloc[0]

    igg_a = next(a for a in ANALYTES if a.col == 'igg')
    iga_a = next(a for a in ANALYTES if a.col == 'iga')
    igm_a = next(a for a in ANALYTES if a.col == 'igm')

    igg, iga, igm = row['igg'].iloc[0], row['iga'].iloc[0], row['igm'].iloc[0]

    very_safe = (
    (row['final_prediction'].iloc[0] == 0) and
    (row['oligoclonal_probability'].iloc[0] < 0.2) and
    (row['comment_108'].iloc[0] < 0.2)
    )

    # Klassificera varje immunglobulin
    levels = {}
    for analyte, value, key in [(igg_a, igg, 'igg'), (iga_a, iga, 'iga'), (igm_a, igm, 'igm')]:
        low, high = get_ref(analyte, gender)
        alarm = ALARM_THRESHOLDS[key]
        levels[key] = classify_ig(value, low, high, alarm['low'], alarm['high'])

    # Specialfall IgA-brist
    if levels['igg'] == 'normal' and levels['igm'] == 'normal' and iga <= 0.07:
        return "IgA-halt <0.07 g/L inger misstanke om medfödd selektiv IgA-brist. "

    all_normal   = all(v == 'normal' for v in levels.values())
    any_elevated = any(v in ['förhöjd', 'lätt förhöjd'] for v in levels.values())
    any_lowered  = any(v in ['sänkt', 'lätt sänkt'] for v in levels.values())
    polyklonal   = " med polyklonal immunglobulinfördelning" if very_safe and not any_lowered else ""

    # Gruppera förhöjda
    for severity in ['förhöjd', 'lätt förhöjd']:
        if severity == 'förhöjd':
            names = [a.name for a, k in [(igg_a,'igg'),(iga_a,'iga'),(igm_a,'igm')]
                     if levels[k] == 'förhöjd']
        else:
            names = [a.name for a, k in [(igg_a,'igg'),(iga_a,'iga'),(igm_a,'igm')]
                     if levels[k] == 'lätt förhöjd' and
                     not any(levels[k2] == 'förhöjd' for k2 in ['igg','iga','igm'])]

        if len(names) == 1:
            comment += f"{'Förhöjd' if severity == 'förhöjd' else 'Lätt förhöjd'} halt av {names[0]}{polyklonal}. "
        elif len(names) > 1:
            comment += f"{'Förhöjda' if severity == 'förhöjd' else 'Lätt förhöjda'} halter av {' och '.join(names)}{polyklonal}. "

    # Gruppera sänkta
    for severity in ['sänkt', 'lätt sänkt']:
        if severity == 'sänkt':
            names = [a.name for a, k in [(igg_a,'igg'),(iga_a,'iga'),(igm_a,'igm')]
                     if levels[k] == 'sänkt']
        else:
            names = [a.name for a, k in [(igg_a,'igg'),(iga_a,'iga'),(igm_a,'igm')]
                     if levels[k] == 'lätt sänkt' and
                     not any(levels[k2] == 'sänkt' for k2 in ['igg','iga','igm'])]

        if len(names) == 1:
            comment += f"{'Sänkt' if severity == 'sänkt' else 'Lätt sänkt'} halt av {names[0]}. "
        elif len(names) > 1:
            comment += f"{'Sänkta' if severity == 'sänkt' else 'Lätt sänkta'} halter av {' och '.join(names)}. "


    # Normalt
    if all_normal and very_safe:
        comment += "Immunglobuliner med normala halter och polyklonal fördelning. "
    elif all_normal:
        comment += "Immunglobuliner med normala halter. "

    return comment

def comment_add_urine(row) -> str:
    igg_a = next(a for a in ANALYTES if a.col == 'igg')
    iga_a = next(a for a in ANALYTES if a.col == 'iga')
    igm_a = next(a for a in ANALYTES if a.col == 'igm')
    gender = row['gender'][0]
    igg, iga, igm = row['igg'].iloc[0], row['iga'].iloc[0], row['igm'].iloc[0]

    u_krea = row['u_kreatinin'].iloc[0]
    has_urine = u_krea is not None and u_krea > 0

    if (igg < get_ref(igg_a, gender)[0] or igm < get_ref(igm_a, gender)[0]) and not has_urine:
        return "Är orsaken till immunglobulin-sänkningen ej känd rekommenderas analys av FLC eller U-proteinprofil för att utesluta Bence-Jones proteinuri. "
    
    return ""