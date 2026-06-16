from bottle import route, run, request, response
import pandas as pd
import json
import traceback
import numpy as np

# Importera dina AI-modeller och textfunktioner från ditt befintliga skript
# (Ändra 'my_ml_script' till namnet på din huvudfil)
from functions.full_model import (
    predict, 
    comment_m_component, 
)

from functions.other_proteins import comment_add_urine
from functions.other_proteins import comment_albumin
from functions.other_proteins import comment_antitrypsin
from functions.other_proteins import comment_haptoglobin
from functions.other_proteins import comment_immunglobulin
from networks.inflammation_network import comment_inflammation
from functions.free_light_chains import comment_free_light_chains
from functions.urine import comment_urin
from functions.process_data import find_proportions
from functions.analyte import ANALYTES

# ── HJÄLPMETODER FÖR BERÄKNINGAR ─────────────────────────────────────────────


def generate_text_interpretation(df_row: pd.DataFrame) -> str:
    """Bygger ihop det medicinska textutlåtandet baserat på din logik."""
    # Säkra upp orders-kolumnen som en tom lista till att börja med
    df_row["orders"] = [[]]
    df_row["orders"] = df_row["orders"].astype(object)

    interpretation = ""
    interpretation += comment_albumin(df_row)
    interpretation += comment_inflammation(df_row)
    interpretation += comment_haptoglobin(df_row)
    interpretation += comment_antitrypsin(df_row)
    interpretation += comment_immunglobulin(df_row)
    interpretation += comment_m_component(df_row)
    interpretation += comment_free_light_chains(df_row)
    interpretation += comment_add_urine(df_row)
    interpretation += comment_urin(df_row)
    
    return interpretation.strip()

# ── API ROUTE ────────────────────────────────────────────────────────────────

@route('/api/analyze', method='POST')
def analyze_patient():
    # Tvinga Bottle att svara med UTF-8 och JSON-header
    response.content_type = 'application/json; charset=utf-8'
    
    try:
        # 1. Ta emot rådatan från JSON-anropet
        raw_data = request.json
        if not raw_data:
            # Fallback om klienten råkar skicka som rå-textsträng
            raw_data = json.loads(request.body.read().decode('utf-8'))
        
        # 2. Konvertera indatan till en Pandas DataFrame (en rad)
        # ── BYGG DATAFRAME KORREKT ───────────────────────────────────────
        # Skalära kolumner hanteras direkt, array-kolumner wrappas i lista
        scalar_cols = {k: v for k, v in raw_data.items() 
                       if not isinstance(v, list)}
        
        

        
        df = pd.DataFrame([scalar_cols])  # En rad med alla skalära värden

        for  a in ANALYTES:
            val = raw_data.get(a.col)
            df[a.col] = np.nan if val is None else np.float32(val)

        # Array-kolumner: varje cell ska vara en numpy-array, INTE en kolumn per element
        df['value']      = pd.array([np.array(raw_data['value'],      dtype=np.float32)], dtype=object)
        df['boundaries'] = pd.array([np.array([], dtype=np.float32)], dtype=object)  # fylls av find_proportions
        df['fractions']  = pd.array([np.array([], dtype=np.float32)], dtype=object)  # fylls av find_proportions

        # ── find_proportions returnerar en Series (iloc[0]) → konvertera tillbaka till DF
        df = find_proportions(df.iloc[0])   # returnerar Series
        if isinstance(df, pd.Series):
            df = df.to_frame().T.reset_index(drop=True)

        # Konvertera till numpy EFTER find_proportions
        for col in ['value', 'boundaries', 'fractions']:
            val = df.at[0, col]
            if not isinstance(val, np.ndarray):
                df.at[0, col] = np.array(val, dtype=np.float32)

        # ── ÅTERSTÄLL ARRAY-KOLUMNER EFTER find_proportions ──────────────
        # find_proportions kan ha skrivit över dessa – sätt rätt värden explicit
        df.at[0, 'value']      = np.array(raw_data['value'],      dtype=np.float32)
        df.at[0, 'boundaries'] = np.array(raw_data['boundaries'], dtype=np.float32) \
                                  if 'boundaries' in raw_data else df.at[0, 'boundaries']
        df.at[0, 'fractions']  = np.array(raw_data['fractions'],  dtype=np.float32) \
                                  if 'fractions'  in raw_data else df.at[0, 'fractions']
        
        # 4. Kör hela din AI-pipeline (CNN, AE, Oligo, Inflammation, Comment108)
        df_predicted = predict(df)
        
        # 5. Generera det medicinska textutlåtandet baserat på alla prediktioner
        text_output = generate_text_interpretation(df_predicted)
        
        # 6. Packa ihop det slimmade svaret till klienten
        result = {
            "status": "success",
            "id": int(df_predicted.loc[0, "row_id"]) if "row_id" in df_predicted.columns else None,
            "predictions": {
                "cnn_probability": float(df_predicted.loc[0, "cnn_probability"]),
                "encoder_probability": float(df_predicted.loc[0, "encoder_probability"]),
                "oligoclonal_probability": float(df_predicted.loc[0, "oligoclonal_probability"]),
                "comment_108_probability": float(df_predicted.loc[0, "comment_108"]),
                "final_prediction": int(df_predicted.loc[0, "final_prediction"])
            },
            "utlatande": text_output,
            "bestalda_analyser": df_predicted.loc[0, "orders"]
        }
        
        # ensure_ascii=False behövs för att svenska tecken (å, ä, ö) ska skickas rätt
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        # Om något skiter sig (t.ex. saknade parametrar i JSON), skicka en 400 Bad Request
        response.status = 400
        traceback.print_exc()
        return json.dumps({
            "status": "error", 
            "message": f"Ett fel uppstod vid bearbetningen: {str(e)}"
        }, ensure_ascii=False)

# ── STARTA SERVER ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # host='0.0.0.0' gör att den lyssnar på alla nätverkskort, 
    # så den gamla Macen kan anropa den via din nya dators IP-adress.
    run( debug=True, reloader=True)