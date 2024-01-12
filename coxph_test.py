def coxph_test(data, tE, sE, covariates, percentile):
    """
    Fit a Cox Proportional Hazards model and perform a multivariate log-rank test.

    Parameters:
    - data (pandas.DataFrame): The input data.
    - tE (str): The column name representing the time variable.
    - sE (str): The column name representing the event/censoring variable.
    - covariates (str array): covariates to be considered in multivariate testing
    - percentile (int): 25, 50 or 75
    
    Returns:
    - cph_summary (pandas.DataFrame): Summary of the Cox Proportional Hazards model fit.
    - log-rank test p-value: H0 complete model not better fitting data than null model
    """
    import pandas as pd
    import numpy as np
    from lifelines.datasets import load_rossi
    from lifelines import CoxPHFitter
    from lifelines import KaplanMeierFitter
    from lifelines.plotting import add_at_risk_counts
    from lifelines.statistics import multivariate_logrank_test
    from scipy.stats import chi2
    
    if not isinstance(covariates, list):
        covariates = [covariates]
    
    data = data[[tE,sE] + covariates]
    data = data.dropna()
    data_cox = pd.DataFrame()
    for cv in covariates:
        data_cv = data[~(pd.isna(data[cv]))]
        dtps = data_cv[cv].dtype
        if dtps=="category":
            dummies = pd.get_dummies(data_cv[cv], prefix=cv, dtype=int, drop_first=True)
            dummies.sum(axis=0) < 2
            dummie_df = pd.DataFrame(dummies.loc[:,~(dummies.sum(axis=0) < 2)])
            if (not data_cox.shape[0]==0):
                dummie_df.index = data_cox.index
            data_cox = pd.concat([data_cox, dummie_df], axis=1)
        else:
            if ((dtps=="float")|(dtps=="int")):
                
                cutoff = np.percentile(data_cv[cv], percentile)
                float_dum = np.repeat("low", data_cv.shape[0])
                float_dum = np.where(data_cv[cv] > cutoff, "high", float_dum)
                float_dum = pd.Series(float_dum).astype("category").cat.set_categories(["low","high"])
                dummies = pd.get_dummies(float_dum, prefix=cv, dtype=int, drop_first=True)
                dummie_df = pd.DataFrame(dummies)
                if (not data_cox.shape[0]==0):
                    dummie_df.index = data_cox.index
                data_cox = pd.concat([data_cox, dummie_df], axis=1)
    data_cox = data_cox.astype(int)
    data_cox.index = data.index
    data_EP = pd.concat([data[[tE, sE]], data_cox], axis=1)
    data_EP = data_EP.dropna()
    cph_full = CoxPHFitter()
    cph_full.fit(data_EP, duration_col=tE, event_col=sE)
    log_likelihood_full = cph_full.log_likelihood_
    # Fit a null model without covariates
    cph_null = CoxPHFitter()
    cph_null.fit(data_EP[[tE, sE]], tE, event_col=sE)
    log_likelihood_null = cph_null.log_likelihood_
    # Calculate the test statistic, which follows a chi-squared distribution
    test_statistic = -2 * (log_likelihood_null - log_likelihood_full)
    # Get the degrees of freedom, which is the difference in the number of parameters between the models
    df = len(cph_full.params_) - len(cph_null.params_)
    # Calculate the p-value
    p_value = chi2.sf(test_statistic, df)


    # Fit a null model without covariates
    # Create a combined categorical variable
    # Then use this in your multivariate_logrank_test

    #p_value = result_all_surv._p_value[0]

    cph_summary = cph_full.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    cph_summary["Log rank p"] = np.repeat("", cph_summary.shape[0])
    cph_summary["Log rank p"][0] = p_value
    cph_summary["significance"] = np.repeat("", cph_summary.shape[0])
    
    #significance stars
    def get_sig_stars(p_value):
        if p_value < 0.00005:
            return "*****"
        elif p_value < 0.0005:
            return "****"
        elif p_value < 0.005:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""
    
    cph_summary["significance"][0] = get_sig_stars(p_value)
    cph_summary["endpoint"] = np.repeat("", cph_summary.shape[0])
    cph_summary["endpoint"][0] = tE
    new_row = pd.DataFrame({col: [""] for col in cph_summary.columns}, index=[0])
    cats = cph_summary.index.to_list()
    cats.insert(0,"")
    cph_summary = pd.concat([new_row, cph_summary]).reset_index(drop=True)

    cph_summary.insert(0, 'Variable(s)', "")
    cph_summary["Variable(s)"][0] = ", ".join(covariates)
    cph_summary.insert(1, 'categories', cats)
    cph_summary.index = ['']*cph_summary.shape[0]
    cph_summary = cph_summary.rename(columns={cph_summary.columns[2]: "hazard ratio", cph_summary.columns[3]: "95%-CI low", cph_summary.columns[4]: "95%-CI high", cph_summary.columns[5]: "Wald test p"})
    pval_out = "{:.2e}".format(p_value)
    return cph_summary, pval_out