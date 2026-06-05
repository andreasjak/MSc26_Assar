from functions.analyte import get_ref
from functions.analyte import get_analyte_by_column_name



def comment_urin(df) -> str:
    comment ='Urin: '

    u_albumin_krea = df['u_albumin'][0]
    u_igg_krea = df['u_igg'][0]
    u_kappa_krea = df['u_kappa'][0]
    u_lambda_krea = df['u_lambda'][0]
    u_hc_krea = df['u_hc'][0]
    gender = df['gender'][0]


    total_protein = sum([u_albumin_krea,u_igg_krea,u_kappa_krea,u_lambda_krea,u_hc_krea])

    albumin_krea = get_analyte_by_column_name('u_albumin')
    igg_krea = get_analyte_by_column_name('u_igg')
    kappa_krea = get_analyte_by_column_name('u_kappa')
    lambda_krea = get_analyte_by_column_name('u_lambda')
    hc_krea = get_analyte_by_column_name('u_hc')

    if total_protein < 5: #Kommentar 802
        comment += 'normala proteinhalter i urinen. '

    if 5 <= total_protein < 50:
        comment += 'lätt-måttlig proteinuri. ' #kommentar 805
    if 50 <= total_protein < 300:
        comment += 'kraftig proteinuri. '  #kommentar 806
    if total_protein >= 300:
        comment += 'massiv proteinuri. '   #kommentar 807

    if albumin_krea > get_ref(albumin_krea,gender)[1]: #Kanske bara ta den ifall man har < 50 totalhalt typ?
        comment += 'Albuminuri. ' #Kommentar 810

    if (u_hc_krea > get_ref(hc_krea)[1]):
        if (u_hc_krea > get_ref(hc_krea,gender)[1] and (u_kappa_krea > get_ref(kappa_krea,gender)[1] or u_lambda_krea > get_ref(lambda_krea,gender)[1] )) and (u_albumin_krea < get_ref(albumin_krea,gender)[1]):
            comment += 'Tubulärt proteinmönster i urinen.' #Kommentar 808
        elif (u_lambda_krea == 0 and u_kappa_krea > 20) or (u_lambda_krea > 0 and u_kappa_krea/u_lambda_krea > 20):
            comment += 'Förhöjda halter av protein HC och kappakedjor i urinen.' #815
        elif u_lambda_krea > get_ref(lambda_krea,gender)[1] and u_kappa_krea > get_ref(kappa_krea,gender)[1]:
            comment += 'Förhöjda halter av protein HC, kappa- och lambdakedjor i urinen.'
        else:
            comment += 'Förhöjd halt av protein HC i urinen. '  #Kommentar 813
    else:
        if (u_lambda_krea > 0 and (1 <= u_kappa_krea/u_lambda_krea <= 4) ):
            comment += 'Kvoten U-Kappa/U-Lambda är normal, vilket talar emot förekomst av Bence-Jones proteinuri. ' #831
        elif u_lambda_krea == 0 and u_kappa_krea < 20:
            comment += 'Kvoten U-Kappa/U-Lambda kan ej bestämmas. Bence-Jones proteinuri av låg halt kan ej uteslutas.' #830
    
    if u_lambda_krea == 0 and u_kappa_krea == 0:
        comment += 'Inga hållpunkter för Bence-Jones proteinuri. ' #824
    if (u_lambda_krea == 0 and u_kappa_krea > 20) or (u_lambda_krea == 0 and u_kappa_krea < 5) or (u_lambda_krea != 0 and (u_kappa_krea/u_lambda_krea < 1 or u_kappa_krea/u_lambda_krea > 4)):
        comment += 'Avvikande kappa-lambda-kvot. Urinimmunfiation rekommenderas.'
    
    

    