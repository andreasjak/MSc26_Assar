from functions.analyte import get_analyte_by_column_name, get_ref
def comment_free_light_chains(df) -> str:
    comment = ''

    s_kappa = df['s_kappa'][0]
    s_lambda = df['s_lambda'][0]
    s_kl_kvot = df['s_kl_kvot'][0]
    gender = df['gender'][0]
    prediction = df['prediction'][0]
    if s_kappa is None or s_lambda is None or s_kl_kvot is None: return ''
    kappa = get_analyte_by_column_name('s_kappa')
    lamda = get_analyte_by_column_name('s_lambda')
    kl_kvot = get_analyte_by_column_name('s_kl_kvot')
    print(s_kappa)

    if get_ref(kl_kvot,gender)[0] <= s_kl_kvot <= get_ref(kl_kvot,gender)[1]:
        if s_kappa > get_ref(kappa,gender)[1] or s_lambda > get_ref(lamda,gender)[1]:
            comment += 'Kvoten fria kappa/lambda-kedjor i serum är normal, vilket talar emot monoklonal produktion av fria lätta immunglobulinkedjor.'
        else:
            comment += 'Halten av fria kappa- och lambdakedjor samt kvoten av fria kappa/lambda-kedjor i serum är normala, vilket talar emot monoklonal produktion av fria lätta immunglobulinkedjor.'


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

def predict_using_free_light_chains(s_kl_kvot: float) -> int:
    if 0.31 <= s_kl_kvot <= 1.56: return 0
    return 1