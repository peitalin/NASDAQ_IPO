
import glob, json, os, re
import itertools
import pandas as pd
import numpy as np
import arrow
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.formula.api as smf
import statsmodels as sm

from functools          import partial
from pprint             import pprint
from numpy              import log, median, sign
from widgets            import as_cash, next_row, next_col



IPO_DIR = os.path.join(os.path.expanduser("~"), "Data", "IPO")
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ",)
FILEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "Filings")
FINALJSON = json.loads(open('final_json.txt').read())
FULLJSON = json.loads(open('full_json.txt').read())



if '__tools__':
    def _vlookup_firms(D=FULLJSON):
        ciks_conames = {cik:D[cik]['Company Overview']['Company Name'] for cik in D}
        conames_ciks = {D[cik]['Company Overview']['Company Name']:cik for cik in D}
        return ciks_conames, conames_ciks

    _ciks_conames, _conames_ciks = _vlookup_firms()

    iprint = partial(print, end=' '*32 + '\r')

    def firmname(cik):
        return _ciks_conames[cik]

    def get_cik(firm):
        ciks = [k for k in _conames_ciks if k.lower().startswith(firm.lower())]
        print("Found => {}".format(ciks))
        return _conames_ciks[ciks[0]]

    def aget(sdate):
        sdate = sdate if isinstance(sdate, str) else sdate.isoformat()
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', sdate):
            return arrow.get(sdate, 'M-D-YYYY') if '-' in sdate else arrow.get(sdate, 'M/D/YYYY')
        elif re.search(r'\d{4}[/-]\d{2}[/-]\d{2}', sdate):
            return arrow.get(sdate, 'YYYY-MM-DD') if '-' in sdate else arrow.get(sdate, 'YYYY/MM/DD')

    def print_pricerange(s):
        "arg s: either firmname or cik. Returns price-ranges"

        if not s.isdigit():
            cik = [k for k in FINALJSON if firmname(k).lower().startswith(s.lower())][0]
        else:
            cik = s

        filings = FINALJSON[cik]['Filing']

        print("\n{A}> Filing Price Range: {B}: {C} <{A}".format(A='='*25, B=firmname(cik), C=cik)[:91])
        print("Date\t\tFormtype\tFile\t\t\t\tPrice Range")
        for v in filings:
            filing_ashx = v[-2]
            price_range = v[-1][0]
            if price_range == 'NA':
                NA_filepath = os.path.join(BASEDIR, 'filings', cik, filing_ashx)
                filesize = os.path.getsize(NA_filepath)
                print("{}\t{}\t\t{}\t{}\t<= {:,} kbs".format(v[2], v[1], v[3], v[-1], round(filesize/1000)))
            else:
                print("{}\t{}\t\t{}\t{}".format(v[2], v[1], v[3], v[-1]))
        print("="*90+'\n')

    def abnormal_svi(df, window=15, category='all'):
        return pd.read_csv("IoT/ASVI_{}day_{}.csv".format(window, category), dtype={'cik': object}).set_index('cik')

        """
        ASVI15 = abnormal_svi(df, window=15, category='weighted_finance')
        ASVI30 = abnormal_svi(df, window=30, category='weighted_finance')
        ASVI60 = abnormal_svi(df, window=60, category='weighted_finance')
        """







VNAME = {
    'Intercept': 'Constant',
    'priceupdate_up': 'Price Update (up)',
    'priceupdate_down': 'Price Update (down)',
    'pct_final_revision_up': 'Final Price Revision (up)',
    'pct_final_revision_down': 'Final Price Revision (down)',
    'prange_change_plus': 'Price Range Change',
    'number_of_price_updates': 'No. Price Amendments',
    'delay_in_price_update' : 'Delay in 1st price update',
    'log(days_from_s1_to_listing)': 'log(Days from S-1 to Listing)',
    'underwriter_rank_avg': 'Underwriter Rank',
    'VC': 'Venture Capital',
    'confidential_IPO': 'Confidential IPO',
    'share_overhang': 'Share Overhang',
    'log(proceeds)': 'log(Proceeds)',
    'log(market_cap)': 'log(Market Cap)',
    'log(1 + sales)': 'log(1+Sales)',
    'liab_over_assets': 'Liab/Assets',
    'EPS': 'EPS+',
    'M3_indust_rets': '3-Month Industry Returns',
    'M3_initial_returns': '3-Month Average IPO Returns',
    'CASI': 'CASI',
    'np.square(CASI)': 'CASI^2',
    'I(CASI^2)': 'CASI^2'
}


if __name__=='__main__':
    ciks = sorted(FINALJSON.keys())
    cik = '1439404' # Zynga         # 8.6
    cik = '1418091' # Twitter       # 7.4
    cik = '1271024' # LinkedIn      # 9.6
    cik = '1500435' # GoPro         # 8.1
    cik = '1318605' # Tesla Motors  # 8
    cik = '1326801' # Facebook      # 8.65
    cik = '1564902' # SeaWorld      # 9.54
    cikfb = '1326801' # Facebook
    ciks1 = ['1439404', '1418091', '1271024', '1500435', '1318605', '1594109', '1326801', '1564902']

    iotkeys = ['gtrends_name', 'IoT_entity_type',
    'IoT_15day_CASI_all', 'IoT_30day_CASI_all', 'IoT_60day_CASI_all',
    'IoT_15day_CASI_finance', 'IoT_30day_CASI_finance', 'IoT_60day_CASI_finance',
    'IoT_15day_CASI_business_industrial', 'IoT_30day_CASI_business_industrial',
    'IoT_60day_CASI_business_industrial', 'IoT_15day_CASI_weighted_finance',
    'IoT_30day_CASI_weighted_finance', 'IoT_60day_CASI_weighted_finance']


    amends = {"None": 0, "Up":1, "Down":-1}

    dfu = pd.read_csv("df_update.csv", dtype={'cik':object})
    dfu.set_index("cik", inplace=1)
    dfu['amendment'] = [amends[x] for x in dfu['amends']]
    dfu['Year'] = dfu['Year'].astype(object)

    df = pd.read_csv("df.csv", dtype={'cik':object})
    df.set_index("cik", inplace=1)
    df['amendment'] = [amends[x] for x in df['amends']]
    df['Year'] = df['Year'].astype(object)

    # dup = df[df['prange_change_first_price_update'].notnull()]
    dup = df[df['size_of_first_price_update'].notnull()]
    dnone = df[df['prange_change_first_price_update'].isnull()]





def xls_empirics(lm, column='C', sheet='15dayCASI', cluster=('FF49_industry', 'Year'),sigstars=True):

    from xlwings import Workbook, Range, Sheet
    wb = Workbook("xl_empirics.xlsx")
    Sheet(sheet).activate()
    column = column.upper()

    def roundn(i, n=2):
        coef = str(round(i, n))
        for i in range(n-len(coef.split('.')[-1])):
            coef += '0'
        return coef

    def is_logit(lm):
        return isinstance(lm, sm.discrete.discrete_model.MultinomialResultsWrapper)

    def write_coefs_tvals(varnames, coefs, tvals, pvals, column=column):

        for v, coef, tval, pval in zip(varnames, coefs, tvals, pvals):

            if v.startswith('Year'):
                continue

            if ':' in v:
                v1, v2 = v.split(':')
                v1 = re.sub(r"IoT_\d\dday_(?=CASI_)CASI_[a-zA-Z_]*", 'CASI', v1)
                v2 = re.sub(r"IoT_\d\dday_(?=CASI_)CASI_[a-zA-Z_]*", 'CASI', v2)
                # Renames IOTKEY, np.square(IOTKEY), and log(IOTKEY) only.
                if 'CASI' in v1:
                    v = v1 + ":" + v2
                else:
                    v = v2 + ":" + v1
                Range('B' + VROW[v]).value = VNAME[v1] + " x " + VNAME[v2]
            else:
                v = re.sub(r"IoT_\d\dday_(?=CASI_)CASI_[a-zA-Z_]*", 'CASI', v)
                Range('B' + VROW[v]).value = VNAME[v]

            coef = roundn(coef, n=2)
            if sigstars:
                if pval <= 0.001:
                    coef += '***'
                elif pval <= 0.01:
                    coef += '**'
                elif pval <= 0.05:
                    coef += '*'

            Range(column + VROW[v]).value = coef
            Range(column + str(int(VROW[v])+1)).value = tval


    def rpy_square_terms(eq, Rpy=True):
        """Convert squared covariates syntax between R and Python.
        Rpy=True => Python to R: np.square(X) -> I(X^2).
        RPY=False => R to Python"""

        eq = re.sub(r'\(Intercept\)', 'Intercept', eq)
        if Rpy:
            var = re.findall('(?<=np.square\()[\w_]*(?=\))', eq)
            for v in var:
                eq = eq.replace('np.square({})'.format(v), 'I({}^2)'.format(v))
        else:
            var = re.findall('(?<=I\()[\w_]*(?=\^2\))', eq)
            for v in var:
                eq = eq.replace('I({}^2)'.format(v), 'np.square({})'.format(v))
        return eq


    # Range("C3:J3").value = ['(1)', '(2)', '(3)', '(4)',
    #                         None, '(6)', '(7)', '(8)',]
    Range("{col}4:{col}60".format(col=column)).value = [[None]]*60

    if isinstance(lm, str):
        eq = lm
        if 'np.square' in eq:
            eq = rpy_square_terms(eq)


        r("M <- lm({eq}, data=dfR)".format(eq=eq))
        r("src('clmclx.R')")
        if len(cluster) == 2:
            r("M_cluster <- mclx(M, 1, dfR$%s, dfR$%s)" % cluster)
        else:
            r("M_cluster <- clx(M, 1, dfR$%s)" % cluster)


        varnames = r("names(M$coefficients)".format(eq=eq))
        varnames = [rpy_square_terms(v, Rpy=False) for v in varnames]
        coefs = r("M_cluster[,1]".format(eq=eq))
        pvals = r("M_cluster[,4]".format(eq=eq))
        tvalues = r("M_cluster[,3]".format(eq=eq))
        tvals = ['(%s)' % roundn(t, n=2) for t in tvalues]

        Range('B' + VROW['Nobs']).value = 'No. Obs'
        Range('B' + str(int(VROW['Rsq'])-1)).value = 'R^2'
        Range(column + VROW['Nobs']).value = r("M$df.residual + M$rank")[0]
        Range(column + str(int(VROW['Rsq'])-1)).value = r("summary(M)$r.squared")[0]

        write_coefs_tvals(varnames, coefs, tvals, pvals)

    elif not is_logit(lm):
        varnames = tuple(lm.params.keys())
        rlm = lm.get_robustcov_results()
        coefs = rlm.params
        pvals = rlm.pvalues
        tvals = ['(%s)' % roundn(t, n=2) for t in rlm.tvalues]

        Range('B' + VROW['Nobs']).value = 'No. Obs'
        Range('B' + str(int(VROW['Rsq'])-1)).value = 'R^2'
        Range(column + VROW['Nobs']).value = rlm.nobs
        Range(column + str(int(VROW['Rsq'])-1)).value = rlm.rsquared

        write_coefs_tvals(varnames, coefs, tvals, pvals)

    elif is_logit(lm):
        varnames = tuple(mnl.params.index)
        coefs = mnl.params
        pvals = mnl.pvalues
        tvals = mnl.tvalues

        for k in coefs:
            # k number of choice variables
            col = chr(ord(column) + k)
            print(tvals)
            t_stats = ['(%s)' % roundn(t, n=2) for t in tvals[k]]
            write_coefs_tvals(varnames, coefs[k], t_stats, pvals[k], column=col)

        Range('B' + VROW['Nobs']).value = 'No. Obs'
        Range('B' + str(int(VROW['Rsq'])-1)).value = 'R^2'
        Range(column + VROW['Nobs']).value = mnl.nobs
        Range(column + str(int(VROW['Rsq'])-1)).value = mnl.prsquared






def xls_final_revisions(days=15):


    if '__variable_lookups__':
        ## THIS SETS THE XLS Variable Order
        VARS = [
            'Intercept',
            'CASI',
            'np.square(CASI)',
            'CASI:priceupdate_up',
            'CASI:priceupdate_down',
            'CASI:VC',
            # 'CASI:delay_in_price_update',
            'priceupdate_up',
            'priceupdate_down',
            'log(days_from_s1_to_listing)',
            'number_of_price_updates',
            # 'delay_in_price_update',
            'underwriter_rank_avg',
            'VC',
            'confidential_IPO',
            'share_overhang',
            'log(proceeds)',
            'EPS',
            'M3_indust_rets',
            'M3_initial_returns',
            'Nobs',
            'Rsq',
        ]
        VROW = dict(zip(VARS, [str(n) for n in range(4,100,2)]))

    IOTKEY = 'IoT_{}day_CASI_weighted_finance'.format(days)
    ALLVAR = [
            'Year',
            'log(days_from_s1_to_listing)',
            'number_of_price_updates',
            'underwriter_rank_avg',
            'VC',
            'confidential_IPO',
            'share_overhang',
            'log(proceeds)',
            'EPS',
            'M3_indust_rets',
            'M3_initial_returns',
            'priceupdate_up',
            'priceupdate_down', # 13
            IOTKEY,
            'np.square({})'.format(IOTKEY), # 15
            '{}:{}'.format(IOTKEY, 'priceupdate_up'),
            '{}:{}'.format(IOTKEY, 'priceupdate_down'),
            '{}:{}'.format(IOTKEY, 'VC'), # 18
        ]

    from rpy2.robjects import r
    list(map(r, """
        require(sandwich)
        require(lmtest)
        require(plm)
        source("clmclx.R")
        dfR <- data.table::fread("df.csv", colClasses=c(cik="character", SIC="character", Year="factor"))""".split('\n')))


    for i, col in zip([13,15,18], 'CEF'):
        X = " + ".join(ALLVAR[:i])
        # lm = smf.ols('percent_final_price_revision ~ ' + X, data=df).fit()
        lm = 'percent_final_price_revision ~ ' + X
        xls_empirics(lm, column=col, sheet='{}dayCASI'.format(days))


    # ALL

    IOTKEY = 'IoT_{}day_CASI_all'.format(days)
    ALLVAR = [
            'Year',
            'log(days_from_s1_to_listing)',
            'number_of_price_updates',
            'underwriter_rank_avg',
            'VC',
            'confidential_IPO',
            'share_overhang',
            'log(proceeds)',
            'EPS',
            'M3_indust_rets',
            'M3_initial_returns',
            'priceupdate_up',
            'priceupdate_down', # 13
            IOTKEY,
            'np.square({})'.format(IOTKEY), # 15
            '{}:{}'.format(IOTKEY, 'priceupdate_up'),
            '{}:{}'.format(IOTKEY, 'priceupdate_down'),
            '{}:{}'.format(IOTKEY, 'VC'), # 18
        ]

    for i, col in zip([15,18], 'HI'):
        X = " + ".join(ALLVAR[:i])
        # lm = smf.ols('percent_final_price_revision ~ ' + X, data=df).fit()
        lm = 'percent_final_price_revision ~ ' + X
        xls_empirics(lm, column=col, sheet='{}dayCASI'.format(days))





    """


    15day CASI has larger coefficients than 30 and 60 day CASI. The results from 60 day CASI are economically weak compared to 15day and 30 day CASI regressions (omitted).


    Coefficient estimates of 1.16 for priceupdate suggests incremental predictability of price amendments in determining the final offer price. However including CASi variables attenuates this effect close to 1.


    CASI is robust across various model specifications, and has incremental explanatory power of price updates, which are noted to vary almost 1 to 1 with final price revisions.

    This suggests much information is incorporated into the 1st price amendment and price amendments appear to adjust fully.


    Increase in CASI is associated with an increase in final price revision. Controlling for the first price amendment,
    A one standard deviation in CASI is associated with a X% final price revision.

    The effect of CASI on final price revision diminishes as CASi increases.
    A economically significant -0.46 on squared CASI suggests that IPOs which attract too much abnormal attention can experience lowered price revisions.


    CASI_15day:
    min: -3.59
    mean: 0.49
    std: 1.20
    max: 8.83



    In model (3), differentiating FPR w.r.t CASI gives the estimated incremental effect of CASI on final price revisions. A unit increase in CASI is associated with an increase of:
    3.31 - 2*0.41*CASI

    For the average firm with CASI = 0.5, this is 3.31 - 2*0.41*0.5 = 2.9% final price revision, controlling for price amendments and other IPO related variables.




    Model (4):
    Introducing interaction variables halves the economic significant of CASI as a standalone variable. Statistical significance also suffers, with t-stat reducing from 4.61 to 1.98--marginally significant at the 5% level.


    Interactions:
    CASI and sq(CASI) effects:
        1.53 - 2*0.46 = 0.61

    CASI X Pupdate_UP = 0.25


    Pupdate_UP = 1.02. A 1 percent increase in filing price range is associated with a 1 percent increase in the final offer price. In other words, the first price update appears to be efficiently incorporated in the final offer price.


    The effect of underwriter rank on final price revisions is indistinguishable from 0 across all models. The way IPOs are priced appear to follow a industry standard practice independent on underwriter rank.

    """



###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################



    IOTKEY = 'IoT_{}day_CASI_weighted_finance'.format(days)
    # IOTKEY = 'IoT_{}day_CASI_news'.format(days)
    # IOTKEY = 'IoT_{}day_CASI_all'.format(days)
    ALLVAR = [
            'log(days_from_s1_to_listing)',
            # 'delay_in_price_update',
            'number_of_price_updates',
            'underwriter_rank_avg',
            'VC',
            # 'confidential_IPO',
            'Year',
            # 'FF49_industry',
            'share_overhang',
            'log(proceeds)',
            'EPS',
            'M3_indust_rets',
            # 'M3_initial_returns', # 9
            'priceupdate_up',
            'priceupdate_down',
            IOTKEY,               # 13
            'np.square(%s)' % IOTKEY,
            '{}:{}'.format(IOTKEY, 'priceupdate_up'),
            # 'np.square({}):{}'.format(IOTKEY, 'priceupdate_up'),
            # '{}:{}'.format(IOTKEY, 'priceupdate_down'),
            # '{}:{}'.format(IOTKEY, 'VC'),
            # '{}:{}'.format(IOTKEY, 'delay_in_price_update'),
            # '{}:{}'.format('delay_in_price_update', 'priceupdate_up'),
            # '{}:{}'.format('delay_in_price_update', 'amends'),
            # '%s:%s:%s' % (IOTKEY, 'VC', 'priceupdate_up'),
        ]

    X = " + ".join(ALLVAR)
    lm = smf.ols('percent_final_price_revision ~ ' + X, data=df).fit()
    rlm = lm.get_robustcov_results()
    rlm.summary()



    IOTKEY = 'IoT_{}day_CASI_weighted_finance'.format(days)
    # IOTKEY = 'IoT_{}day_CASI_news'.format(days)
    # IOTKEY = 'IoT_{}day_CASI_all'.format(days)
    ALLVAR = [
            'log(days_from_s1_to_listing)',
            # 'delay_in_price_update',
            'number_of_price_updates',
            'underwriter_rank_avg',
            'VC',
            # 'confidential_IPO',
            'share_overhang',
            'log(proceeds)',
            # 'log(1 + sales)',
            # 'FF49_industry',
            'Year',
            'EPS',
            'M3_indust_rets',
            'M3_initial_returns', # 9
            'pct_final_revision_up',
            'pct_final_revision_down',
            IOTKEY,               # 13
            'np.square(%s)' % IOTKEY,
            '%s:%s' % (IOTKEY, 'pct_final_revision_up'),
            '%s:np.square(%s)' % (IOTKEY, 'pct_final_revision_up'),
            # '%s:%s' % ('underwriter_rank_avg', 'pct_final_revision_up'),
            # '%s:%s:%s' % ('underwriter_rank_avg', 'pct_final_revision_up', IOTKEY),
            # '%s:%s' % (IOTKEY, 'pct_final_revision_down'),
            '%s:%s' % (IOTKEY, 'VC'),
        ]

    X = " + ".join(ALLVAR)
    lm = smf.ols('close_return ~ ' + X, data=df).fit()
    rlm = lm.get_robustcov_results()
    rlm.summary()



    """ interaction between IOTKEY and final_price revision:
    OLS with uncorrected errors is significant,
    Robust errors is insignificant. Why?
    This is due to conditional underpricing, larger price revisions are associated with
    both larger AND more variable returns.
    Hence naively correcting the standrad errors after OLS will always give insignificant results.

    There may be nonlinear market reactions to price revisions in the presence of large abnormal attention.
    Clearly, this implies additional structure needs to be incorporated into the IPO returns model.

    In this section, we specifically model IPO initial returns as a hierarchical(multilevel) model.
    """




    # dup = dfu[dfu['prange_change_first_price_update'].notnull()]
    dup = dfu[dfu['size_of_first_price_update'].notnull()]

    IOTKEY = 'IoT_{}day_CASI_weighted_finance'.format(days)
    ALLVAR = [
            'Year',
            'log(days_from_s1_to_listing)',
            'underwriter_rank_avg',
            'VC',
            'confidential_IPO',
            'share_overhang',
            'log(proceeds)',
            'log(market_cap)',
            'log(1+sales)',
            'liab_over_assets',
            'EPS',
            'M3_indust_rets',
            'M3_initial_returns', # 13
            'delay_in_price_update', # 14
            IOTKEY,               # 13
            'np.square(%s)' % IOTKEY,
            # '%s:%s' % (IOTKEY, 'VC'),
        ]

    X = " + ".join(ALLVAR)
    lm = smf.ols('percent_first_price_update ~ ' + X, data=dup).fit()
    rlm = lm.get_robustcov_results()
    rlm.summary()

    # 0: No amend
    # 1: upwards amend
    # 2: downwards amend
    mnl = smf.mnlogit('amendment ~ ' + X, data=df).fit()
    mnl.summary()

###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################




def mnl_amends():

    # 0: No amend
    # 1: upwards amend
    # -1: downwards amend
    X = """Year + log(days_from_s1_to_listing) + underwriter_rank_avg + VC + confidential_IPO + share_overhang + log(proceeds) + log(market_cap) + log(1+sales) + liab_over_assets + EPS + M3_indust_rets + M3_initial_returns + delay_in_price_update + IoT_15day_CASI_weighted_finance + np.square(IoT_15day_CASI_weighted_finance) + IoT_15day_CASI_weighted_finance:VC
    """
    mnl = smf.mnlogit('amendment ~ ' + X, data=df).fit()
    mnl.summary()
    marg = mnl.get_margeff()
    xls_empirics(mnl, column='C', sheet='MNL_amends'.format(days))





def xls_price_updates():


    ###############################################
    # First price update regression

    """
    delay_in_price_update measures the length of time left to subscribe in the IPO (expressed as a fraction of total time from initial pricing to listing).
    This information is known at all times because the listing date is set ahead of time and common knowledge.

    """


    ## THIS SETS THE XLS Variable Order
    VARS = [
        'Intercept',
        'CASI',
        'np.square(CASI)',
        'CASI:VC',
        'delay_in_price_update',
        'log(days_from_s1_to_listing)',
        'underwriter_rank_avg',
        'VC',
        'confidential_IPO',
        'share_overhang',
        'log(proceeds)',
        'log(market_cap)',
        'log(1 + sales)',
        'liab_over_assets',
        'EPS',
        'M3_indust_rets',
        'M3_initial_returns',
        'Nobs',
        'Rsq',
    ]
    VROW = dict(zip(VARS, [str(n) for n in range(4,100,2)]))



    ###############################################
    # First price update regression

    # dup = dfu[dfu['prange_change_first_price_update'].notnull()]
    dup = dfu[dfu['size_of_first_price_update'].notnull()]


    ALLVAR = [
            'Year',
            'log(days_from_s1_to_listing)',
            'underwriter_rank_avg',
            'VC',
            'confidential_IPO',
            'share_overhang',
            'log(proceeds)',
            'log(market_cap)',
            'log(1+sales)',
            'liab_over_assets',
            'EPS',
            'M3_indust_rets',
            'M3_initial_returns', # 13
            'delay_in_price_update', # 14
        ]


    from rpy2.robjects import r
    list(map(r, """
        require(sandwich)
        require(lmtest)
        require(plm)
        source("clmclx.R")
        df <- data.table::fread("df.csv", colClasses=c(cik="character", SIC="character", Year="factor"))
        dfR <- df[!is.na(df$size_of_first_price_update)]""".split('\n')))



    # Fit controls only
    for i, col in zip([13,14], 'CD'):
        X = " + ".join(ALLVAR[:i])
        lm = 'percent_first_price_update ~ ' + X
        xls_empirics(lm, column=col, sheet='updates_CASI', cluster='FF49_industry')


    models = list(itertools.product([15,30], [2,3]))
    for model, col in zip(models, 'FGIJ'):
        days, i = model
        IOTKEY = 'IoT_{days}day_CASI_weighted_finance'.format(days=days)
        XVAR = [IOTKEY, 'np.square({})'.format(IOTKEY), '{}:{}'.format(IOTKEY, 'VC')]
        X = " + ".join(ALLVAR + XVAR[:i])
        lm = 'percent_first_price_update ~ ' + X
        xls_empirics(lm, column=col, sheet='updates_CASI', cluster='FF49_industry')



    """
    I make sure CASI is consistent with the date of the event, and sum CASI in the D-days before an event of interest (dependent variables such as filing price range amendments, or the final price revision).
    """



    """ I look at a subsample of firms whic update offer prices early.
    Which kind of firms are more likely to amend offer prices upwards?
    It appears that delayed issues are strongly less likely to update prices upwards,
    issues that attract more attention are more likely to experience upwards amendments
    """



    ##############################################
    # Initial Returns regression
    # Interaction: CASI * Final_price_revision
    controls = [
            'Year',
            'pct_final_revision_up',
            'pct_final_revision_down',
            'prange_change_plus',
            'log(days_from_s1_to_listing)',
            # 'days_first_price_change',
            'underwriter_rank_avg',
            'VC',
            'confidential_IPO',
            'share_overhang',
            'log(proceeds)',
            'log_firm_size',
            'liab_over_assets',
            'EPS',
            'M3_indust_rets',
            'M3_initial_returns',
            ]

    IOTKEY = 'IoT_15day_CASI_weighted_finance'
    INTERACT = [
                '{} * {}'.format(IOTKEY,'pct_final_revision_up'),
                '{} * {}'.format(IOTKEY,'pct_final_revision_down'),
                ]
    Y = 'close_return'
    # Y = 'log(Volume)'
    X = " + ".join(controls + [IOTKEY] + INTERACT)

    results = smf.ols(Y + ' ~ ' + X, data=df).fit()
    results.summary()



    # smf.rlm(Y + ' ~ ' + X, data=df).fit().summary()

    """
     In every case, the statistical significance of CASI itself is driven out by the interaction term between CASI and change in offer price.
     The results remain the same whether initial returns is defined as open_returns (Aggarwal et al. 2002) or close_returns (Lowry and Schwert, 2004).
    """


    ## Regularization - LASSO and Ridge regression
    import patsy
    y, x = patsy.dmatrices(Y + " ~ " + X, data=df)
    from sklearn import linear_model

    ridge = linear_model.Ridge(alpha=0.5)
    ridge.fit(x,y)
    list(zip(['intercept']+X.split('+'), ridge.coef_[0]))


    lasso = linear_model.Lasso(alpha=0.5)
    lasso.fit(x,y)
    list(zip(['intercept']+X.split('+'), lasso.coef_))

    """ LASSO Coefs as a function of regularization
    alphas = np.linspace(0,2,100)
    coefs = []
    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(x,y)
        coefs.append(lasso.coef_)

    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()
    """



    # # Price update regression
    # controls = [
    #          'underwriter_rank_avg',
    #          'VC', 'confidential_IPO', 'NASDAQ',
    #          'share_overhang',
    #          'log(proceeds)',
    #          'log_firm_size',
    #          'liab_over_assets',
    #          'EPS',
    #          'M3_indust_rets',
    #          # 'M3_IPO_volume', # <- multicollinearity issuers with indust rets
    #          'M3_initial_returns',
    #      ]


    # IOTKEY = 'IoT_15day_CASI_weighted_finance'
    # INTERACT = [
    #             # '{} * {}'.format(IOTKEY,'M3_IPO_volume')
    #             ]
    # Y = 'percent_first_price_update'
    # X = " + ".join(controls + [IOTKEY] + INTERACT)

    # results = smf.ols(Y + ' ~ ' + X, data=dfu).fit()
    # results.summary()






    # CASI up to 1st update
    # dfu = pd.read_csv("DFU.csv", dtype={'cik':object})
    # dfu.set_index("cik", inplace=1)





    # dfu['percent_final_price_revision'] *= 100
    # dfu['percent_first_price_update'] *= 100
    # dfu['M3_indust_rets'] *= 100
    # dfu['M3_initial_returns'] *= 100
    # dfu['priceupdate_down'] *= 100
    # dfu['priceupdate_up'] *= 100
    # dfu['close_return'] *= 100
    # dfu['open_return'] *= 100
    # dfu['pct_final_revision_down'] *= 100
    # dfu['pct_final_revision_up'] *= 100


    # CASI up to final price revision
    # df = pd.read_csv("df.csv", dtype={'cik':object})
    # df.set_index("cik", inplace=1)

    # df['percent_final_price_revision'] *= 100
    # df['percent_first_price_update'] *= 100
    # df['M3_indust_rets'] *= 100
    # df['M3_initial_returns'] *= 100
    # df['priceupdate_down'] *= 100
    # df['priceupdate_up'] *= 100
    # df['close_return'] *= 100
    # df['open_return'] *= 100
    # df['pct_final_revision_down'] *= 100
    # df['pct_final_revision_up'] *= 100

