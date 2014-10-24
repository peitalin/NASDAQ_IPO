
import glob, json, os, re
import pandas as pd
import numpy as np
import requests
import arrow
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.formula.api as smf

from functools          import partial
from collections        import Counter, OrderedDict
from pprint             import pprint
from concurrent.futures import ThreadPoolExecutor
from numpy              import log, median, sign

from IPython            import embed
from widgets            import as_cash, write_FINALJSON

FILE_PATH = 'text_files/'
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

        print("\n{A}> Filing Price Range: {B}: {C} <{A}".format(A='='*22, B=firmname(cik), C=cik))
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


def plot_iot(cik, category=''):
    "plots Google Trends Interest-over-time (IoT)"

    import matplotlib.dates as mdates

    def gtrends_file(cik, category):
        gtrends_dir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'gtrends-beta', 'cik-ipo', category)
        # gtrends_dir = os.path.join(BASEDIR, 'cik-ipo', category)
        return os.path.join(gtrends_dir, str(cik)+'.csv')

    cik = str(cik) if not isinstance(cik, str) else cik
    cik = get_cik(cik) if not cik.isdigit() else cik
    # colors = [rgb_to_hex(c) for c in sb.color_palette("colorblind")]
    colors = ['#0072b2', '#009e73', '#d55e00', '#cc79a7', '#f0e442', '#56b4e9']


    if category == '':
        iot_data_fin = pd.read_csv(gtrends_file(cik=cik, category='finance'))
        iot_data_all = pd.read_csv(gtrends_file(cik=cik, category='all'))
        if len(iot_data_fin) == len(iot_data_all):
            iot_data = iot_data_fin
            category = 'finance'
        else:
            iot_data = iot_data_all
            category = 'all'
    else:
        iot_data = pd.read_csv(gtrends_file(cik=cik, category=category))


    iot_data = pd.read_csv(gtrends_file(cik=cik, category=category))
    firm = iot_data.columns[1]
    iot_data['Date'] = [arrow.get(d).date() for d in iot_data['Date']]
    # iot_melt = pd.melt(iot_data.icol([0,1]), id_vars=['Date'])
    # iot_melt.columns = ['Date', firm, 'Interest-over-time']
    # ax = iot_melt.groupby([firm, 'Date']).mean().unstack('Tesla Motors').plot()

    s1_date = arrow.get(df.loc[cik, 'date_s1_filing']).date()
    anno_index1 = iot_data[iot_data.Date == s1_date].index[0]

    roadshow_date = arrow.get(df.loc[cik, 'date_1st_pricing']).date()
    anno_index2 = iot_data[iot_data.Date == roadshow_date].index[0]

    date_listed = arrow.get(df.loc[cik, 'date_trading']).date()
    anno_index3 = iot_data[iot_data.Date == date_listed].index[0]


    fig, ax = plt.subplots(sharex=True, figsize=(15,5))
    ax.plot(iot_data["Date"], iot_data[firm], label='Search Interest: {} ({})'.format(firm, iot_data.columns[2]))
    ax.annotate('S-1 Filing',
                (mdates.date2num(iot_data.Date[anno_index1]), iot_data[firm][anno_index1]),
                xytext=(20, 20),
                size=11,
                color=colors[2],
                textcoords='offset points',
                arrowprops=dict(width=1.5, headwidth=5, shrink=0.1, color=colors[2]))

    ax.annotate('Roadshow Begins',
                (mdates.date2num(iot_data.Date[anno_index2]), iot_data[firm][anno_index2]),
                xytext=(15, 15),
                size=11,
                color=colors[2],
                textcoords='offset points',
                arrowprops=dict(width=1.5, headwidth=5, shrink=0.1, color=colors[2]))

    ax.annotate('IPO Listing Date',
                (mdates.date2num(iot_data.Date[anno_index3]), iot_data[firm][anno_index3]),
                xytext=(20, 20),
                size=11,
                color=colors[2],
                textcoords='offset points',
                arrowprops=dict(width=1.5, headwidth=5, shrink=0.1, color=colors[2]))

    plt.title("Interest-over-time for {} - ({})".format(firm, category))
    plt.ylabel("Search Interest")
    plt.legend()
    plt.show()


def abnormal_svi(df, window=15, category='all'):
    return pd.read_csv("IoT/ASVI_{}day_{}.csv".format(window, category), dtype={'cik': object}).set_index('cik')


"""

ASVI15 = abnormal_svi(df, window=15, category='weighted_finance')
ASVI30 = abnormal_svi(df, window=30, category='weighted_finance')
ASVI60 = abnormal_svi(df, window=60, category='weighted_finance')

"""




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

    iotkeys = ['gtrends_name', 'IoT_entity_type', 'IoT_15day_CASI_all', 'IoT_30day_CASI_all', 'IoT_60day_CASI_all',
                'IoT_15day_CASI_finance', 'IoT_30day_CASI_finance', 'IoT_60day_CASI_finance',
                'IoT_15day_CASI_business_industrial', 'IoT_30day_CASI_business_industrial', 'IoT_60day_CASI_business_industrial',
                'IoT_15day_CASI_weighted_finance', 'IoT_30day_CASI_weighted_finance', 'IoT_60day_CASI_weighted_finance']


    # CASI up to 1st update
    dfu = pd.read_csv("DFU.csv", dtype={'cik':object})
    dfu.set_index("cik", inplace=1)

    """
    dfu['NASDAQ'] = ['NASDAQ' in x for x in dfu['exchange']]
    dfu['log_firm_size'] = log(dfu['market_cap'])
    dfu['percent_final_price_revision'] *= 100
    dfu['percent_first_price_update'] *= 100
    dfu['M3_indust_rets'] *= 100
    dfu['M3_IPO_volume'] *= 100
    dfu['M3_initial_returns'] *= 100
    dfu['priceupdate_down'] *= 100
    dfu['priceupdate_up'] *= 100
    dfu['close_return'] *= 100
    dfu['open_return'] *= 100
    dfu['prange_change'] = [x if x==x else 0 for x in dfu['prange_change_first_price_update']]
    """

    # CASI up to final price revision
    df = pd.read_csv("DF.csv", dtype={'cik':object})
    df.set_index("cik", inplace=1)

    """
    df['NASDAQ'] = ['NASDAQ' in x for x in df['exchange']]
    df['log_firm_size'] = log(df['market_cap'])
    df['percent_final_price_revision'] *= 100
    df['percent_first_price_update'] *= 100
    df['M3_indust_rets'] *= 100
    df['M3_IPO_volume'] *= 100
    df['M3_initial_returns'] *= 100
    df['priceupdate_down'] *= 100
    df['priceupdate_up'] *= 100
    df['close_return'] *= 100
    df['open_return'] *= 100
    df['prange_change'] = [x if x==x else 0 for x in df['prange_change_first_price_update']]
    df['pct_final_revision_up'] = [x if x > 0 else 0 for x in df['percent_final_price_revision']]
    df['pct_final_revision_down'] = [x if x < 0 else 0 for x in df['percent_final_price_revision']]
    """



    # Price revision regression
    controls = [
             'priceupdate_down',
             'priceupdate_up',
             'prange_change',
             'log(days_s1_to_listing)',
             # 'days_first_price_change',
             # 'underwriter_collective_rank',
             'underwriter_rank_avg',
             'VC',
             'confidential_IPO',
             # 'NASDAQ',
             'share_overhang',
             'log_proceeds',
             'log_firm_size',
             'liab_over_assets',
             'EPS',
             'M3_indust_rets',
             # 'M3_IPO_volume', # <- multicollinearity issuers with indust rets
             'M3_initial_returns',
             # 'IoT_30day_CASI_weighted_finance',
             # 'IoT_60day_CASI_weighted_finance',
         ]


    IOTKEY = 'IoT_15day_CASI_weighted_finance'
    INTERACT = [
                '{} * {}'.format(IOTKEY,'priceupdate_up'),
                '{} * {}'.format(IOTKEY,'priceupdate_down'),
                # '{} * {}'.format(IOTKEY,'log(days_s1_to_listing)'),
                ]
    Y = 'percent_final_price_revision'
    X = " + ".join(controls + [IOTKEY] + INTERACT)

    lm = smf.ols(Y + ' ~ ' + X, data=df).fit()
    lm.summary()

    smf.rlm(Y + ' ~ ' + X, data=df).fit().summary()

    """
    as expected, 15day CASI has larger coefficients than 30 and 60 day CASI.
    CASI is robust across various model specifications, and has incremental explanatory power of price updates, which are noted to vary almost 1 to 1 with final price revisions.

    # Model results are sensitive to addition of log_firm_size, although CASI remains economically significant.

    Multicollinearity ??
    df[controls + iotkeys].corr()
    """



    ###############################################
    # Initial Returns regression
    # Interaction: CASI * Price updates
    controls = [
             'priceupdate_down',
             'priceupdate_up',
             'prange_change',
             'log(days_s1_to_listing)',
             'days_first_price_change',
             # 'underwriter_collective_rank',
             'underwriter_rank_avg',
             'VC',
             'confidential_IPO',
             # 'NASDAQ',
             'share_overhang',
             'log_proceeds',
             'log_firm_size',
             # 'liab_over_assets',
             'EPS',
             'M3_indust_rets',
             # 'M3_IPO_volume', # <- multicollinearity issuers with indust rets
             'M3_initial_returns',
             # 'IoT_30day_CASI_weighted_finance',
             # 'IoT_60day_CASI_weighted_finance',
         ]
    IOTKEY = 'IoT_15day_CASI_weighted_finance'
    INTERACT = [
                '{} * {}'.format(IOTKEY,'priceupdate_down'),
                '{} * {}'.format(IOTKEY,'priceupdate_up'),

                ]
    Y = 'close_return'
    # Y = 'open_return'
    X = " + ".join(controls + [IOTKEY] + INTERACT)

    results = smf.ols(Y + ' ~ ' + X, data=df).fit()
    results.summary()

    ##############################################
    # Initial Returns regression
    # Interaction: CASI * Final_price_revision
    controls = [
             # 'priceupdate_down',
             # 'priceupdate_up',
             'prange_change',
             # 'log(days_s1_to_listing)',
             'days_first_price_change',
             # 'underwriter_collective_rank',
             'underwriter_rank_avg',
             'VC',
             'confidential_IPO',
             # 'NASDAQ',
             'share_overhang',
             'log_proceeds',
             'log_firm_size',
             # 'liab_over_assets',
             'EPS',
             'M3_indust_rets',
             # 'M3_IPO_volume', # <- multicollinearity issuers with indust rets
             'M3_initial_returns',
             # 'IoT_30day_CASI_weighted_finance',
             # 'IoT_60day_CASI_weighted_finance',
         ]

    IOTKEY = 'IoT_30day_CASI_weighted_finance'
    INTERACT = [
                '{} * {}'.format(IOTKEY,'percent_final_price_revision'),
                ]
    Y = 'close_return'
    # Y = 'open_return'
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
    #          'log_proceeds',
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





