from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from sklearn.model_selection import train_test_split
from analyte import ANALYTES


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
    
    # Bygg mapping från analysnummer -> värde
    mapping = {}
    urine_nrs = ['0039','0033','0034','0035','0036']
    for analysis, value in zip(analysis_list, value_list):
        if analysis in urine_nrs and str(value).strip() == 'KOMM':
            value = 0.0
        else:
            value = clean_value(value)
        mapping[analysis.replace('*', '')] = value
    
    # Överskrid med total-värden om de finns
    for col, nr in [('total_igg', '0064'), ('total_iga', '0065'), ('total_igm', '0066')]:
        val = clean_value(row.get(col, None))
        if val is not None and not np.isnan(val):
            mapping[nr] = val
    
    # Sätt värden för alla analyter
    for analyte in ANALYTES:
        row[analyte.col] = mapping.get(analyte.analysis_nr, np.nan)
    
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
    if str(val).strip() == 'NEG':
        return 0.0  # eller np.nan om du vill exkludera
    try:
        return float(str(val).replace('<', '').replace('>', '').strip())
    except:
        return np.nan
    
def split_sets(df: pd.DataFrame) -> pd.DataFrame:
    pids = df['pid'].unique()
    valtrain_pids, test_pids = train_test_split(pids, test_size=0.10,random_state=15)
    _, val_pids = train_test_split(valtrain_pids, test_size=0.05,random_state=15)
    df['set'] = 'train'  # default
    df.loc[df['pid'].isin(val_pids), 'set'] = 'val'
    df.loc[df['pid'].isin(test_pids), 'set'] = 'test'
    return df
    
def pre_process_data(df: pd.DataFrame) -> pd.DataFrame:
    rows = split_sets(df)
    rows = rows[ rows['value'].apply(len).isin([300,301]) ] #de flesta har en onödig datapunkt
    rows['value'] = rows['value'].apply(lambda x: x[:300]) #ta bort onödiga datapunkten
    rows['age'] = rows['age'].apply(parse_age)
    rows = extract_protein_values(rows) 
    rows = rows.apply(find_proportions, axis=1)
    return rows

    
