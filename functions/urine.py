from functions.analyte import get_ref
from functions.analyte import get_analyte_by_column_name
import numpy as np


def comment_urin(df) -> str:

    u_albumin_krea = df['u_albumin_krea'][0]
    u_igg_krea = df['u_igg_krea'][0]
    u_kappa_krea = df['u_kappa_krea'][0]
    u_lambda_krea = df['u_lambda_krea'][0]
    u_hc_krea = df['u_hc_krea'][0]
    gender = df['gender'][0]
    u_kappa = df['u_kappa'][0]
    u_lambda = df['u_lambda'][0]

    if np.isnan([u_albumin_krea,u_igg_krea,u_kappa_krea,u_lambda_krea,u_hc_krea,u_kappa,u_lambda]).any():
        return ''

    comment ='Urin: '


    total_protein = sum([u_albumin_krea,u_igg_krea,u_kappa_krea,u_lambda_krea,u_hc_krea])

    albumin_krea = get_analyte_by_column_name('u_albumin_krea')
    igg_krea = get_analyte_by_column_name('u_igg_krea')
    kappa_krea = get_analyte_by_column_name('u_kappa_krea')
    lambda_krea = get_analyte_by_column_name('u_lambda_krea')
    hc_krea = get_analyte_by_column_name('u_hc_krea')



    if u_albumin_krea < get_ref(albumin_krea,gender)[1] and u_igg_krea < get_ref(igg_krea,gender)[1] and u_kappa_krea < get_ref(kappa_krea,gender)[1] and u_lambda_krea < get_ref(lambda_krea,gender)[1] and u_hc_krea < get_ref(hc_krea,gender)[1]: #Kommentar 802
        comment += 'normala proteinhalter i urinen. '
    if 5 <= total_protein < 50:
        comment += 'lätt-måttlig proteinuri. ' #kommentar 805
    if 50 <= total_protein < 300:
        comment += 'kraftig proteinuri. '  #kommentar 806
    if total_protein >= 300:
        comment += 'massiv proteinuri. '   #kommentar 807

    if u_albumin_krea >= get_ref(albumin_krea,gender)[1] and u_hc_krea < get_ref(hc_krea,gender)[1] and u_kappa_krea < get_ref(kappa_krea,gender)[1] and u_lambda_krea < get_ref(lambda_krea,gender)[1] and total_protein < 5 + albumin_krea - 3:
        comment += 'Albuminuri. ' #Kommentar 810

    if u_albumin_krea > 30 and u_hc_krea < get_ref(hc_krea,gender)[1] and u_kappa_krea < get_ref(kappa_krea,gender)[1] and u_lambda_krea < get_ref(lambda_krea,gender)[1] and total_protein < 5 + u_albumin_krea - 3: 
        comment += 'Påtaglig albuminuri. ' #Kommentar 811
    
    if u_albumin_krea > 60 and u_hc_krea < get_ref(hc_krea,gender)[1] and u_kappa_krea < get_ref(kappa_krea,gender)[1] and u_lambda_krea < get_ref(lambda_krea,gender)[1] and total_protein < 5 + u_albumin_krea - 3: 
        comment += 'Kraftig albuminuri. ' #Kommentar 811

    if (u_hc_krea >= get_ref(hc_krea,gender)[1] and u_albumin_krea < get_ref(albumin_krea,gender)[1]):
        if  u_lambda_krea >= get_ref(lambda_krea,gender)[1] and u_kappa_krea >= get_ref(kappa_krea,gender)[1]:
            comment += 'Tubulärt proteinmönster i urinen. '
        elif (u_lambda_krea == 0 and u_kappa_krea >= get_ref(kappa_krea,gender)[1]):
            comment += 'Förhöjda halter av protein HC och kappakedjor i urinen. ' #815
        else:
            comment += 'Förhöjd halt av protein HC i urinen. '  #Kommentar 813
    
    if (u_lambda > 0 and (1 < u_kappa/u_lambda < 4) ):
        comment += 'Kvoten U-Kappa/U-Lambda är normal, vilket talar emot förekomst av Bence-Jones proteinuri. ' #831

    elif u_lambda == 0 and  0 < u_kappa < 20:
        comment += 'Kvoten U-Kappa/U-Lambda kan ej bestämmas. Bence-Jones proteinuri av låg halt kan ej uteslutas. ' #830
    if u_lambda == 0 and u_kappa == 0:
        comment += 'Inga hållpunkter för Bence-Jones proteinuri. ' #824
    if (u_lambda == 0 and u_kappa >= 20) or (u_kappa == 0 and u_lambda > 0) or (u_lambda != 0 and (u_kappa/u_lambda <= 1 or u_kappa/u_lambda >= 4)):
        comment += 'Avvikande kappa-lambda-kvot. '
        current_orders = df.at[0, "orders"]
        if u_lambda < 40 and u_kappa < 40:
            df.at[0, "orders"] = current_orders + ["1713"]
        else:
            df.at[0, "orders"] = current_orders + ["1712"]
    return comment
    

    