from dataclasses import dataclass

@dataclass
class Analyte:
    name: str           # visningsnamn
    col: str            # kolumnnamn i DataFrame
    analysis_nr: str    # analysnummer
    ref_male: tuple
    ref_female: tuple
    unit: str = "g/L"

ANALYTES = [
    Analyte('Albumin',      'albumin',      '0054', (36.0, 45.0), (36.0, 45.0)),
    Analyte('Antitrypsin',  'antitrypsin',  '0056', (0.86, 1.75), (0.94, 1.94)),
    Analyte('Orosomukoid',  'orosomukoid',  '0055', (0.52, 1.17), (0.52, 1.17)),
    Analyte('Haptoglobin',  'haptoglobin',  '0058', (0.24, 1.90), (0.24, 1.90)),
    Analyte('CRP',          'crp',          '0062', (0.0,  3.0),  (0.0,  3.0)),
    Analyte('IGG',          'igg',          '0064', (6.7,  14.5), (6.7,  14.5)),
    Analyte('IGA',          'iga',          '0065', (0.88, 4.5),  (0.88, 4.5)),
    Analyte('IGM',          'igm',          '0066', (0.27, 2.10), (0.27, 2.10)),
    Analyte('S-Kappa',      's_kappa',      '1613', (6.7,  22.4),  (6.7,   22.4)),
    Analyte('S-Lambda',     's_lambda',     '1614', (8.3,  27.0),  (8.3,   27.0)),
    Analyte('S-KL-kvot',    's_kl_kvot',    '1615', (0.31,  1.56),  (0.31,   1.56)),
    Analyte('U-Albumin/krea',    'u_albumin',   '0039', (0.0,  3.0),  (0.0,   3.0)),
    Analyte('U-IGG/krea',    'u_igg',    '0033', (0.0,  0.8),  (0.0,   0.8)),
    Analyte('U-Kappa/krea',    'u_kappa',    '0034', (0.0,  0.6),  (0,   0.6)),
    Analyte('U-Lambda/krea',    'u_lambda',    '0035', (0.0,  0.6),  (0.0,   0.6)),
    Analyte('U-HC/krea',    'u_hc',    '0036', (0.0,  1.6),  (0.0,   1.6)),
]

# Härleds automatiskt från ANALYTES
ANALYSIS_MAP = {a.analysis_nr: a.col for a in ANALYTES}
PROTEIN_LABELS = [a.name for a in ANALYTES]

def get_analyte(name: str) -> Analyte:
    for analyte in ANALYTES:
        if name.lower() == analyte.name.lower(): return analyte
    return None

def get_analyte_by_column_name(col_name: str) -> Analyte:
    for analyte in ANALYTES:
        if col_name.lower() == analyte.col: return analyte
    return None

def get_analyte_by_analysis_nr(analysis_nr: str) -> Analyte:
    for analyte in ANALYTES:
        if analysis_nr == analyte.analysis_nr: return analyte
    return None

def get_refs(gender: str) -> list[tuple]:
    return [a.ref_male if gender == 'M' else a.ref_female for a in ANALYTES]

def get_ref(analyte: Analyte,gender: str) -> list[tuple]:
    if gender == 'M': return analyte.ref_male
    else: return analyte.ref_female
