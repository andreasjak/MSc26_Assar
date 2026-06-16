from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from sklearn.model_selection import train_test_split
from functions.analyte import ANALYTES
from functions.analyte import get_analyte_by_analysis_nr
from functions.analyte import get_all_analyte_ids


def find_proportions(row):
    values = row['value']
    peaks, properties = find_peaks(values,prominence=6,distance=10)
    top6_idx = np.argsort(properties['prominences'])[-6:]
    peaks = np.sort(peaks[top6_idx])  # sortera i höjd-ordning
    widths, heights, left_ips, right_ips = peak_widths(values, peaks, rel_height=0.8)
    left_ips = left_ips.astype(int)
    right_ips = right_ips.astype(int)
    combined = []
    for (left,right) in zip(left_ips,right_ips):
        combined.append(int(left))
        combined.append(int(right))
    row['boundaries'] = combined

    values = np.array(values, dtype=float)
    total_area = np.trapezoid(values)
    boundaries = [0]
    for i in range(len(peaks) - 1):
        # hitta minimum mellan topparna = naturlig dalgång
        valley = np.argmin(values[peaks[i]:peaks[i+1]]) + peaks[i]
        boundaries.append(valley)
    boundaries.append(len(values))
    
    fractions = []
    for i in range(len(peaks)):
        area = np.trapezoid(values[boundaries[i]:boundaries[i+1]])
        fractions.append(float(100 * area / total_area))
    
    row['fractions'] = fractions
    return row


def analyse_proteins(row):
    value_list = row['protein_value']
    analysis_list = row['analysis']

    invalid_proteins = []
    relevant_proteins = get_all_analyte_ids()
    ig_nrs = ['0064', '0065', '0066']  # IgG, IgA, IgM
    
    # Bygg mapping från analysnummer -> värde
    mapping = {}
    for analysis, value in zip(analysis_list, value_list):
        clean_nr = analysis.replace('*', '')
        parsed_value = 0.0
        if clean_nr not in relevant_proteins:
            continue
        if '<'in str(value) or str(value).strip() == 'KOMM' or str(value).strip() == 'NEG':
            if clean_nr not in ig_nrs:
                protein_name = get_analyte_by_analysis_nr(clean_nr).name
                invalid_proteins.append(protein_name)
        else:
            parsed_value = clean_value(value)
    
        mapping[clean_nr] = parsed_value
    
    # Överskrid med total-värden om de finns
    for col, nr in [('total_igg', '0064'), ('total_iga', '0065'), ('total_igm', '0066')]:
        val = clean_value(row.get(col, None))
        if val is not None and not np.isnan(val):
            mapping[nr] = val
    
    # Sätt värden för alla analyter
    for analyte in ANALYTES:
        row[analyte.col] = mapping.get(analyte.analysis_nr, np.nan)
    

    protein_comment = ''
    if len(invalid_proteins) == 1:
        protein_comment = f'{invalid_proteins[0]} är inte mätbart. '

    if len(invalid_proteins) == 2:
        protein_comment = f'{invalid_proteins[0]} och {invalid_proteins[1]} är inte mätbara. '
    
    if len(invalid_proteins) > 2:
        for i in range(len(invalid_proteins) - 1):
            protein_comment += f'{invalid_proteins[i]}, '
        protein_comment += f'och {invalid_proteins[-1]} är inte mätbara. '

    row['protein_comments'] = protein_comment
    return row

def extract_protein_values(rows: pd.DataFrame) -> pd.DataFrame:
    # Initiera alla kolumner med NaN
    for analyte in ANALYTES:
        rows[analyte.col] = np.nan
    
    rows = rows.apply(analyse_proteins, axis=1)
    return rows

def parse_age(s):
    match = re.match(r"(\d+)y(\d+)m", s)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months/12
    return None

def clean_value(val: str) -> float:
    if val is None or val == '' or val == 'NA' or val == 'KOMM':
        return np.nan
    if str(val).strip() == 'NEG' or str(val).strip() == 'KOMM':
        return 0.0  # eller np.nan om du vill exkludera
    try:
        return float(str(val).replace('<', '').replace('>', '').strip())
    except:
        return np.nan
    
def split_sets(df: pd.DataFrame) -> pd.DataFrame:
    pids = np.sort(df['pid'].unique())
    valtrain_pids, test_pids = train_test_split(pids, test_size=0.10,random_state=15)
    _, val_pids = train_test_split(valtrain_pids, test_size=0.05,random_state=15)
    df['set'] = 'train'  # default
    df.loc[df['pid'].isin(val_pids), 'set'] = 'val'
    df.loc[df['pid'].isin(test_pids), 'set'] = 'test'
    return df
    
def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df[ df['value'].apply(len).isin([300,301]) ] #de flesta har en onödig datapunkt
    df['value'] = df['value'].apply(lambda x: x[:300]) #ta bort onödiga datapunkten
    df['age'] = df['age'].apply(parse_age)
    df = extract_protein_values(df) 
    df = df.apply(find_proportions, axis=1)

    df = df[
    (df['fractions'].apply(len) == 6) &
    (df['boundaries'].apply(len) == 12)
    ]
    df = split_sets(df)

    return df

    
