from functions.analyte import get_analyte_by_column_name, get_ref
import numpy as np
import joblib

def comment_free_light_chains(df) -> str:
    comment = ''

    s_kappa = df['s_kappa'][0]
    s_lambda = df['s_lambda'][0]
    s_kl_kvot = df['s_kl_kvot'][0]
    gender = df['gender'][0]
    prediction = df['final_prediction'][0]
    if s_kappa is None or s_lambda is None or s_kl_kvot is None: return ''
    kappa = get_analyte_by_column_name('s_kappa')
    lamda = get_analyte_by_column_name('s_lambda')
    kl_kvot = get_analyte_by_column_name('s_kl_kvot')
    #print(s_kappa)

    if get_ref(kl_kvot,gender)[0] <= s_kl_kvot <= get_ref(kl_kvot,gender)[1]:
        if s_kappa > get_ref(kappa,gender)[1] or s_lambda > get_ref(lamda,gender)[1]:
            comment += 'Kvoten fria kappa/lambda-kedjor i serum är normal, vilket talar emot monoklonal produktion av fria lätta immunglobulinkedjor. '
        else:
            comment += 'Halten av fria kappa- och lambdakedjor samt kvoten av fria kappa/lambda-kedjor i serum är normala, vilket talar emot monoklonal produktion av fria lätta immunglobulinkedjor. '


    if s_kl_kvot > 10 and s_kappa > 100:
        comment += 'Halten av fria kappakedjor och kvoten fria kappa/lambda-kedjor i serum är kraftigt förhöjda, vilket starkt talar för monoklonal produktion av fria kappakedjor. '
        if prediction == 0:
            comment += "Immunfixation rekommenderas. "
    elif s_kl_kvot > get_ref(kl_kvot,gender)[1] and s_kappa > get_ref(kappa,gender)[1]:
        comment += 'Halten av fria kappakedjor och kvoten fria kappa/lambda-kedjor i serum är förhöjda, vilket talar för monoklonal produktion av fria kappakedjor. '
        if prediction == 0:
            comment += "Immunfixation rekommenderas. "

    if s_kl_kvot < 0.05 and s_lambda > 100:
        comment += 'Halten av fria lambdakedjor är kraftigt förhöjd och kvoten fria kappa/lambda-kedjor i serum är kraftigt sänkt, vilket starkt talar för monoklonal produktion av fria lambdakedjor. '
        if prediction == 0:
            comment += "Immunfixation rekommenderas. "
    elif s_kl_kvot < get_ref(kl_kvot,gender)[0] and s_lambda > get_ref(lamda,gender)[1]:
        comment += 'Halten av fria lambdakedjor är förhöjd och kvoten fria kappa/lambda-kedjor i serum är sänkt, vilket talar för monoklonal produktion av fria lambdakedjor. '
        if prediction == 0:
            comment += "Immunfixation rekommenderas. "

    if (s_kl_kvot < 0.31 or s_kl_kvot > 1.56) and len(comment) == 0:
        comment += " Avvikande kvot av fria lätta kedjor i serum. "
        if prediction == 0:
            comment += "Immunfixation rekommenderas. "
    return comment

def predict_using_free_light_chains(df):
    if 's_kl_kvot' not in df.columns:
        return df
    
    cols = ['s_kl_kvot', 's_kappa', 's_lambda', 'cnn_probability']
    mask = df[cols].notna().all(axis=1)
    if mask.sum() == 0:          # ← detta saknades
        df['free_light_chain_flag'] = np.nan
        return df
    
    positive_deviation = np.maximum(0, df.loc[mask, 's_kl_kvot'] - 1.56)
    negative_deviation = np.maximum(0, 0.31 - df.loc[mask, 's_kl_kvot'])
    diff = np.abs(df.loc[mask, 's_kappa'] - df.loc[mask, 's_lambda'])
    prob = df.loc[mask, 'cnn_probability']
    
    X = np.column_stack([positive_deviation, negative_deviation, diff, prob])
    model = joblib.load('../models/free_light_chains.pkl')
    probs = model.predict_proba(X)[:, 1]
    
    kvot = df.loc[mask, 's_kl_kvot'].values
    outside_ref = (kvot < 0.31) | (kvot > 1.56)
    low_prob = probs < 0.1
    
    flags = np.full(mask.sum(), np.nan)
    flags[low_prob] = 0       # Regel 2 först
    flags[outside_ref] = 1   # Regel 1 trumfar

    df['free_light_chain_flag'] = np.nan
    df.loc[mask, 'free_light_chain_flag'] = flags
    return df
