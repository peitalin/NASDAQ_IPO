

import glob, json, os, re
import pandas as pd
import numpy as np
import requests
import arrow
import matplotlib.pyplot as plt
import seaborn as sb

from widgets            import as_cash, write_FINALJSON
from collections        import Counter, OrderedDict
from pprint             import pprint
from concurrent.futures import ThreadPoolExecutor
from IPython            import embed



FILE_PATH = 'text_files/'
IPO_DIR = os.path.join(os.path.expanduser("~"), "Data", "IPO")
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ",)
FILEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "Filings")
FINALJSON = json.loads(open('final_json.txt').read())

aget = lambda x: arrow.get(x, 'M/D/YYYY')
conames_ciks = {cik:FINALJSON[cik]['Company Overview']['Company Name'] for cik in FINALJSON}
firmname = lambda cik: conames_ciks[cik]
get_cik = lambda firm: [x[0] for x in conames_ciks.items() if x[1].lower().startswith(firm)][0]


# None of these functions return a DF, they modify existing df.
def df_filings():
    """Takes FINALJSON dataframes, gets price revision stats"""

    def first_price_update(dates, prices, pranges):
        for i, p in enumerate(prices):
            if p == prices[0]:
                continue
            diff_dates  = (dates[i]-dates[0]).days
            diff_prices = (prices[i]-prices[0])
            diff_prange = (pranges[i]-pranges[0])
            percent_price_change = diff_prices / prices[0]
            return diff_dates, diff_prices, percent_price_change, diff_prange
        return [None, None, None, None]

    def num_price_updates(prices):
        return len([x for x,y in zip(prices, prices[1:]) if x!=y])

    def midpoint(cash_strings):
        prices = [as_cash(s) for s in cash_strings if as_cash(s)]
        if len(cash_strings) > 2:
            raise(ValueError("Not a price range, too many prices: %s" % cash_strings))
        return sum(prices) / len(cash_strings)

    def prange(cash_strings):
        prices = [as_cash(s) for s in cash_strings if as_cash(s)]
        return max(prices) - min(prices)

    def is_cash(string):
        return isinstance(as_cash(string), float)

    float_columns = ['days_to_first_price_update',
                    'days_to_final_price_revision',
                    'days_from_priced_to_listing',
                    'size_of_first_price_update',
                    'size_of_final_price_revision',
                    'percent_first_price_update',
                    'percent_final_price_revision',
                    'prange_change_first_price_update',
                    'number_of_price_updates']
    df = pd.DataFrame([[0]*len(float_columns)]*len(filings), filings.index, columns=float_columns)


    RW, MISSING, ABNORMAL_DURATION = [], [], set()
    # cik, values = '1326801', filings.loc['1326801']
    # cik, values = '1472595', filings.loc['1472595']
    print('Calculating dates and price updates...')
    for cik, values in filings.iterrows():
        if any(l[1] == 'RW' for l in values if l):
            RW.append(cik)
            continue

        values = [v for v in values if v]
        values = [v for v in values if is_cash(v[-1][0]) and as_cash(v[-1][0]) > 3]
        if not values:
            MISSING.append(cik)
            print("{}: {} no filings above 3$".format(cik, firmname(cik)))
            # Minimum NASDAQ listing price is 4$
            continue

        pranges = [prange(v[-1]) for v in values]
        prices = [midpoint(v[-1]) for v in values]
        dates  = [aget(v[2]).date() for v in values]
        dates, prices, pranges = zip(*sorted(zip(dates, prices, pranges)))

        trade_date = arrow.get(open_prices.loc[cik, 'Date']).date()
        B424_date = aget(values[0][2]).date()
        priced_date = aget(company.loc[cik, 'Status']).date()
        s1_date = aget([x for x in filings.loc[cik] if x][-1][2]).date()

        if (trade_date - priced_date).days > 91:
            # trading_date - priced_date > 1: => Firm floats with delay
            if (trade_date - B424_date).days > 91:
                # 424B_date - priced_date: delayed final prospectus
                print("{}: {} =>\n\tTrade date: {}\n\t424B Date: {}\n\tPriced: {}".format(cik, firmname(cik), trade_date, B424_date, priced_date))
                ABNORMAL_DURATION |= {cik}

        # NASDAQ 'priced_date' tends to be early.
        # Get final revision price date
        if priced_date < B424_date < trade_date:
            final_rev_date = B424_date
        elif B424_date < priced_date  < trade_date:
            final_rev_date = priced_date
        else:
            final_rev_date = trade_date

        listing_price = as_cash(company.loc[cik, 'Share Price'])
        filing_price_range = [as_cash(s) for s in values[-1][-1] if as_cash(s)]

        if len(filing_price_range) < 2:
            if filing_price_range[0] - 1 > listing_price:
                outside_range = "under"
            elif filing_price_range[0] + 1 < listing_price:
                outside_range = "above"
            else:
                outside_range = "within"
        else:
            if min(filing_price_range) > listing_price:
                outside_range = "under"
            elif max(filing_price_range) < listing_price:
                outside_range = "above"
            else:
                outside_range = "within"
        offer_in_filing_price_range = outside_range

        if len(prices) == 1:
            days_to_first_price_update = None
            size_of_first_price_update = None
            percent_first_price_update = None
            prange_change_first_price_update = None
            number_of_price_updates = None
            days_to_final_price_revision = None
            size_of_final_price_revision = None
            percent_final_price_revision = None
            # print("{}: {} has 1 filing".format(cik, firmname(cik)))

        elif len(prices) > 1:
            if dates[1] == final_rev_date:
                days_to_first_price_update = None
                size_of_first_price_update = None
                percent_first_price_update = None
            else:
                days_to_first_price_update, \
                size_of_first_price_update, \
                percent_first_price_update, \
                prange_change_first_price_update = first_price_update(dates,prices,pranges)

                if size_of_first_price_update:
                    if size_of_first_price_update > 10:
                        ABNORMAL_DURATION |= {cik}
                        print("{}: {} has large price updates".format(cik, firmname(cik)))

            number_of_price_updates = num_price_updates(prices)
            days_to_final_price_revision = (final_rev_date - dates[0]).days
            size_of_final_price_revision = listing_price - prices[0]
            percent_final_price_revision = (listing_price - prices[0]) / prices[0]

        if days_to_first_price_update is not None:
            if days_to_first_price_update > days_to_final_price_revision:
                days_to_first_price_update = days_to_final_price_revision

        df.loc[cik, 'days_to_first_price_update'] = days_to_first_price_update
        df.loc[cik, 'size_of_first_price_update'] = size_of_first_price_update
        df.loc[cik, 'percent_first_price_update'] = percent_first_price_update

        df.loc[cik, 'days_to_final_price_revision'] = days_to_final_price_revision
        df.loc[cik, 'size_of_final_price_revision'] = size_of_final_price_revision
        df.loc[cik, 'percent_final_price_revision'] = percent_final_price_revision

        df.loc[cik, 'days_from_priced_to_listing'] = (trade_date - dates[0]).days
        df.loc[cik, 'offer_in_filing_price_range'] = offer_in_filing_price_range
        df.loc[cik, 'prange_change_first_price_update'] = prange_change_first_price_update
        df.loc[cik, 'number_of_price_updates'] = number_of_price_updates
        df.loc[cik, 'shares_offered'] = int(company.loc[cik, 'Shares Offered'].replace(',',''))

        df.loc[cik, 'date_1st_pricing'] = dates[0]
        df.loc[cik, 'date_last_pricing'] = priced_date
        df.loc[cik, 'date_424b'] = B424_date
        df.loc[cik, 'date_trading'] = trade_date
        df.loc[cik, 'date_s1_filing'] = s1_date

        file_types = [x[1] for x in filings.loc[cik] if x]
        df.loc[cik, 'foreign'] = 1 if any(f.startswith('F') for f in file_types) else 0


    df['Year'] = [int(2000)]*len(df)
    for cik in df.index:
        df.loc[cik, 'Year'] = aget(FINALJSON[cik]['Company Overview']['Status']).year
    df[float_columns] = df[float_columns].astype(float)
    df['cik'] = df.index
    df.to_csv("df_temp.csv", dtype={'cik':object})

    return df


def order_df(df):
    "Corrects dtypes and reorders keys in dataframe"

    cols = [
        'days_to_first_price_update',
        'days_to_final_price_revision',
        'days_from_priced_to_listing',
        'size_of_first_price_update',
        'size_of_final_price_revision',
        'percent_first_price_update',
        'percent_final_price_revision',
        'prange_change_first_price_update',
        'number_of_price_updates',
        'offer_in_filing_price_range',
        'Coname', 'Year',
        'date_1st_pricing',  'date_last_pricing', 'date_s1_filing', 'date_424b', 'date_trading',
        'Date', 'Offer Price', 'Open', 'Close', 'High', 'Low', 'Volume',
        'open_return', 'close_return',
        'SIC', 'FF49_industry', '2month_indust_rets', '2month_IPO_volume', '2month_pct_above_midpoint', '2month_initial_returns', 'BAA_yield_changes',
        'confidential_IPO', 'foreign', 'exchange',
        'underwriter_rank_avg', 'underwriter_num_leads', 'underwriter_collective_rank', 'underwriter_tier',
        'share_overhang', 'shares_offered', 'total_dual_class_shares', 'log_proceeds',
        'market_cap', 'liab/assets', 'P/E', 'P/sales', 'EPS'
    ]
    missed_cols = sorted(set(df.keys()) - set(cols))

    col_float = ['days_to_first_price_update', 'days_to_final_price_revision', 'days_from_priced_to_listing', 'size_of_first_price_update', 'size_of_final_price_revision', 'percent_first_price_update', 'percent_final_price_revision', 'number_of_price_updates', 'prange_change_first_price_update', 'Offer Price', 'Open', 'Close', 'High', 'Low', 'Volume', 'open_return', 'close_return', '2month_indust_rets', '2month_pct_above_midpoint', '2month_initial_returns', 'BAA_yield_changes', 'underwriter_rank_avg', 'underwriter_collective_rank', 'share_overhang', 'shares_offered', 'log_proceeds', 'market_cap', 'liab/assets', 'P/E', 'P/sales', 'EPS', ]
    col_int = ['Year', 'confidential_IPO', 'underwriter_num_leads', '2month_IPO_volume', 'foreign', 'total_dual_class_shares']
    col_obj = ['offer_in_filing_price_range', 'Coname', 'date_1st_pricing', 'date_last_pricing', 'date_424b', 'date_trading', 'date_s1_filing', 'Date', 'SIC', 'FF49_industry', 'underwriter_tier', 'exchange']

    if not isinstance(df['SIC'].iloc[0], str):
        df["SIC"] = ['0'+str(i) if i<1000 else str(i) for i in df["SIC"]]
    df[col_float] = df[col_float].astype(float)
    df[col_int] = df[col_int].astype(int)
    df[col_obj] = df[col_obj].astype(object)

    return df[cols + missed_cols]


def underwriter_ranks(df):
    """Gets the Ritter/Carter-Manaster rank for the 'Lead Underwriters' in the supplied dataframe
    dataframe musthave keys: lead_underwriter and s1_date
    """

    from functools import partial
    from arrow     import Arrow
    from fuzzywuzzy import process
    uw_rank = pd.read_csv(IPO_DIR + "/SEC_index/uw_rank.csv", encoding="latin-1")
    uw_rank = uw_rank.dropna()
    uw_rank = uw_rank.set_index("Underwriter")
    na_uw = {'No Underwriter', 'Self-underwritten', ' -- '}

    def is_same_alphabet(a_reference, a):
        return a.lower().startswith(a_reference)

    def lookup_underwriter(uw):
        "Matches UW with UR from carter manaster ranks by fuzzystring matching."
        is_same_alpha = partial(is_same_alphabet, uw[0].lower())
        uw_with_same_alpha = set(filter(is_same_alpha, uw_rank.index))
        matched_uw = process.extractOne(uw, uw_with_same_alpha)
        if matched_uw[1] < 80:
            print(uw, '->', matched_uw)
        return matched_uw[0]

    def lookup_rank_year(cik):
        pricing_date = FINALJSON[cik]['Company Overview']['Status']
        if aget(pricing_date) < aget('01/01/2007'):
            return '2005-2007'
        elif aget(pricing_date) > aget('01/01/2010'):
            return '2010-2011'
        else:
            return '2008-2009'

    def get_uw_rank(uw,cik):
        return uw_rank.loc[lookup_underwriter(uw), lookup_rank_year(cik)]

    # Fuzzy match to Carter-Manaster underwriter names
    for cik, uw_list in experts['Lead Underwriter'].items():
        # Filter no underwriter
        if uw_list != uw_list:
            df.loc[cik, 'underwriter_rank_avg'] = -1
            df.loc[cik, 'underwriter_num_leads'] = 0
            df.loc[cik, 'underwriter_collective_rank'] = -1
            continue
        if any(uw in na_uw for uw in uw_list):
            df.loc[cik, 'underwriter_rank_avg'] = 0
            df.loc[cik, 'underwriter_num_leads'] = 0
            df.loc[cik, 'underwriter_collective_rank'] = 0
            continue

        print("Getting underwriter rank for {}: {}".format(cik, firmname(cik)))
        uw_ranks = [get_uw_rank(uw,cik) for uw in uw_list]
        # #Average underwriter rank
        CM_rank = round(sum(uw_ranks)/len(uw_list), 1)
        df.loc[cik, 'underwriter_rank_avg'] = CM_rank
        df.loc[cik, 'underwriter_num_leads'] = len(uw_list)
        df.loc[cik, 'underwriter_collective_rank'] = round(sum(uw_ranks), 1)
    return df


def match_sic_indust_FF(df):
    """Matches df['SIC'] with the industry categories listed in
    one of Kenneth French's industry portfolios.
    """

    import pandas.io.data as web
    import io
    from zipfile import ZipFile

    def build_sic_dict(sic_list):
        "Builds SIC dict: 'Insur': [6011, 6022 ... ]"
        ind = {}
        for line in sic_list:
            if not line:
                continue
            line = line.strip()
            if not line[:4].isdigit():
                sic_key = re.sub(r'^\d*\s*', '', line).split(' ')[0]
                if sic_key != '':
                    ind[sic_key] = []
            else:
                i1, i2 = map(int, line[:9].split('-'))
                sic_codes = range(i1, i2+1)
                sic_codes = ['0'+str(i) if len(str(i)) < 4 else str(i) for i in sic_codes]
                ind[sic_key] += list(map(str, sic_codes))
        return ind

    zf = ZipFile(io.BytesIO(requests.get('http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes49.zip').content))
    sic_list = zf.read(zf.namelist()[0]).decode('latin-1').split('\n')
    sic_dict = build_sic_dict(sic_list)

    if not isinstance(df.SIC.iloc[0], str):
        df["SIC"] = ['0'+str(i) if i<1000 else str(i) for i in df["SIC"]]
    for cik, sic in df["SIC"].items():
        if sic != sic:
            raise(ValueError("SIC is a NaN"))
        indKey = [k for k in sic_dict if sic in sic_dict[k]]
        if indKey:
            df.loc[cik, 'FF49_industry'] = indKey[0]
            print(sic, '->' , indKey[0])
        else:
            print(sic, '-> Other')
            df.loc[cik, 'FF49_industry'] = 'Other'
    return df


def ipo_cycle_variables(df, lag=2, date_key='date_1st_pricing', weighting='value'):
    """Gets FF49 industry returns, and IPO deal volume
        for each firm in <df> for the <lag> months leading up to an IPO.
    """

    import pandas.io.data as web
    import re, arrow, io, requests
    from zipfile import ZipFile
    from collections import Counter

    def FF_industry_returns(filing_dates, lag):
        "Indust Returns (Matched on Ken French's industry portfolios)"

        print("Retrieving Fama-French 49 industry portfolios...")

        # try:
        #     FF49 = pd.read_csv("data/49FF_{}weighted.csv".format(weighting))
        # except OSError:
        #     print("Couldn't find a local copy in ./data/*, getting data from Ken French's website")
        #     FF49 = web.DataReader("49_Industry_Portfolios_daily", "famafrench")
        #     FF49 = FF49[3] if weighting=='value' else FF49[5]
        # finally:
        #     FFkeys = ['Date'] + [re.sub(r'[0-9]{1,2}\s[b]', '', s).replace("'", '') for s in FF49.keys()][1:]
        #     FF49.columns = FFkeys
        #     FF49.set_index('Date', inplace=True)

        FF49 = web.DataReader("49_Industry_Portfolios", "famafrench")[4]
        FFkeys = [re.sub(r'[0-9]{1,2}\s[b]', '', s).replace("'", '') for s in FF49.keys()]
        FF49.columns = FFkeys
        FF49.index = [arrow.get(str(d), 'YYYYMM') for d in FF49.index]
        try:
            FFindustries = df['FF49_industry']
        except:
            raise(KeyError("FF49 industry key missing, run match_sic_indust_FF first."))

        print("Matching industry returns on 'FF49_industry'...")
        df_returns = []
        for industry, date in zip(FFindustries, filing_dates):
            if date not in FF49.index:
                date = FF49.index[-1]
                lag = 1
            industry_returns = FF49[industry]
            date_range = arrow.Arrow.range('month', date.replace(months=(-lag+1)), date)
            df_returns += [sum(industry_returns[d]/100 for d in date_range)/lag]
        return df_returns

    def BAA_yield_spread(filing_dates):
        """Calculates changes in the BAA corporate bond yield spread (Moody's)
        -1 month from df['Date']
        """

        start = sorted(filing_dates)[0].date()
        end = sorted(filing_dates)[-1].date()
        print("Grabbing Moody's BAA corporate yield bond data from FRED...")
        BAA = web.DataReader('BAA', 'fred', start, end)
        BAA.index = [arrow.get(s) for s in BAA.index]

        BAA_yield_changes = []
        for d in filing_dates:
            date_range = arrow.Arrow.range('month', d.replace(months=(-lag+1)), d)
            BAA_lagged = BAA[BAA.index.isin(date_range)]
            BAA_yield_changes += [BAA_lagged['BAA'][-1] - BAA_lagged['BAA'][0]]
        return BAA_yield_changes

    def IPO_market_trends(filing_dates, lag):
        """From Ritter's IPO website, Gets:
        1) avg_1st_day_return on IPOs in <lag> months
        2) net_volume of IPOs (excludes penny stocks, units, closed-end funds, etc)
        3) Percentage of deals that priced above the midpoint of the original file price range
        """
        average = lambda iterable: np.mean([float(d) for d in iterable])

        ipo_cycles = pd.read_csv("data/IPO_cycles.csv")
        ipo_cycles['date'] = [arrow.get(d, 'D/MM/YYYY') for d in ipo_cycles['date']]
        ipo_cycles.set_index('date', inplace=True)

        print("Getting IPO market trends metrics: net_volume, avg_initial_returns, pct_above_midpoint")
        net_volume = []
        avg_initial_return = []
        percent_above_midpoint = []
        for d in filing_dates:
            date_range = arrow.Arrow.range('month', d.replace(months=(-lag+1)), d)

            net_volume.append(
                average(ipo_cycles.loc[d]['net_volume'] for d in date_range))

            avg_initial_return.append(
                average(ipo_cycles.loc[d]['avg_1day_return'] for d in date_range))

            percent_above_midpoint.append(
                average(ipo_cycles.loc[d]['pct_above_midpoint_price'] for d in date_range))

        return net_volume, avg_initial_return, percent_above_midpoint


    filing_dates = [arrow.get(d, 'YYYY-MM') for d in df[date_key]]
    try:
        cycle_keys = [x for x in df.keys() if x[0].isdigit()]
        print("Clearing keys: {}".format(cycle_keys))
        df = df.drop(cycle_keys, axis=1)
    except ValueError:
        pass

    df['{n}month_indust_rets'.format(n=lag)] = FF_industry_returns(filing_dates, lag)
    df['{n}month_IPO_volume'.format(n=lag)], \
    df['{n}month_initial_returns'.format(n=lag)], \
    df['{n}month_pct_above_midpoint'.format(n=lag)] = IPO_market_trends(filing_dates, lag)
    df['BAA_yield_changes'] = BAA_yield_spread(filing_dates)

    return df


def dual_class_shares(df):

    def as_num(string):
        string = re.sub(r'(,|\s--\s)', '', string)
        if string:
            return int(string)

    dual_class_shares = pd.read_excel("data/dualclassIPOs19802014.xls")
    dual_class_shares.set_index("CUSIP", inplace=True)

    for cik in FINALJSON:
        shares_nasdaq = as_num(FINALJSON[cik]['Company Overview']['Shares Outstanding'])
        CUSIP = FINALJSON[cik]['Metadata']['CUSIP']
        if CUSIP in dual_class_shares.index:
            if shares_nasdaq != dual_class_shares.loc[CUSIP, "Shares"]:
                print('Firm: {} => \n\tNasdaq reports: {} shares\n\tDual Class reports: {} shares'
                        .format(firmname(cik), shares_nasdaq, dual_class_shares.loc[CUSIP, "Shares"]))
            df.loc[cik, 'total_dual_class_shares'] = dual_class_shares.loc[CUSIP, "Shares"]
        else:
            df.loc[cik, 'total_dual_class_shares'] = shares_nasdaq
    return df


def share_overhang(df):
    as_int = lambda s: float(s.replace(',', ''))
    for cik in df.index:
        print("Share overhang for: {}".format(cik), end=' '*20+'\r')
        shares_outstanding = df.loc[cik, 'total_dual_class_shares']
        shares_offered = FINALJSON[cik]['Company Overview']['Shares Offered']
        if shares_outstanding==' -- ' or shares_offered==' -- ':
            df.loc[cik, 'share_overhang'] = None
        else:
            if isinstance(shares_outstanding, str):
                shares_outstanding = as_int(shares_outstanding)
            if isinstance(shares_offered, str):
                shares_offered = as_int(shares_offered)
            df.loc[cik, 'share_overhang'] =  shares_outstanding / shares_offered
    return df


def confidential_IPO(df):
    "Checks DRS filings for confidential IPOs"
    print("Reading master.idx for formtypes...")
    master = pd.read_csv("/Users/peitalin/Data/IPO/NASDAQ/data/master-2005-2014.idx",
                    encoding='latin-1', iterator=True).read()
    DRS = master[master['Form Type']=='DRS']
    DRS['CIK'] = [str(int(i)) for i in DRS['CIK']]
    DRS.set_index('CIK', inplace=True)
    drs_cik = set(df.index) & set(DRS.index)

    for cik in df.index:
        if cik in drs_cik:
            df.loc[cik, 'confidential_IPO'] = True
        else:
            df.loc[cik, 'confidential_IPO'] = False
    return df


def financials(df):
    "Populate df with financials metrics: liabilities/assets, EPS, log_proceeds"

    for cik in df.index:
        print("Getting financial ratios for {}: {}".format(cik, firmname(cik)), end=" "*20+'\r')
        FJ = FINALJSON[cik]

        net_income = as_cash(FJ['Financials']['Net Income'])
        revenue = as_cash(FJ['Financials']['Revenue'])
        liabil = as_cash(FJ['Financials']['Total Liabilities'])
        assets = as_cash(FJ['Financials']['Total Assets'])
        proceeds = as_cash(FJ['Company Overview']['Offer Amount'])
        share_price = as_cash(FJ['Company Overview']['Share Price'])
        num_shares = df.loc[cik, 'total_dual_class_shares']

        if num_shares != ' -- ':
            num_shares = float(num_shares)
            market_cap = num_shares * share_price
        else:
            num_shares = None
            market_cap = None

        df.loc[cik, 'market_cap']   = market_cap
        df.loc[cik, 'log_proceeds'] = np.log(proceeds)

        df.loc[cik, 'liab/assets'] = liabil/assets if liabil and assets else None
        df.loc[cik, 'P/E'] = market_cap/net_income if net_income and market_cap else None
        df.loc[cik, 'P/sales'] = market_cap/revenue if revenue and market_cap else None
        df.loc[cik, 'EPS'] = net_income/num_shares if net_income and num_shares else None
        float_cols = ['liab/assets', 'P/E', 'P/sales', 'EPS']
        df[float_cols] = df[float_cols].astype(float)
    return df


def from_FINALJSON_to_df():
    # df = df_filings()
    df = pd.read_csv('df.csv', dtype={'cik':object})
    df.set_index('cik', inplace=True)
    #########################
    df.sort_index(inplace=True)
    df = df.join(open_prices.sort_index())
    cols = ['Open', 'Close', 'High', 'Low', 'Volume']
    df[cols] = df[cols].astype(float)
    df['Offer Price'] = [as_cash(s) for s in company.sort_index()['Share Price']]
    df['open_return'] = (df['Open'] - df['Offer Price']) / df['Offer Price']
    df['close_return'] = (df['Close'] - df['Offer Price']) / df['Offer Price']
    df["SIC"] = metadata["SIC"]
    df['Coname'] = company['Company Name']
    ########
    df = underwriter_ranks(df)
    df = match_sic_indust_FF(df)
    df = share_overhang(df)
    df = confidential_IPO(df)
    df = ipo_cycle_variables(df)
    df = financials(df)
    df = order_df(df)
    df.to_csv("df.csv", dtype={'cik':object})



# GVKEY,IID,Date,Ticker,CUSIP,Coname,Volume,Close,High,Low,Open,NAICS,SIC,CIK







## IoI
def cumulative_iot(df, category):
    """Calculates median interest over time (IoT) 60 days before roadshow date,
    CASI (cumulative abnormal search interest) and CSI (cumulative search interest)

    Args:
        --category: 'business-industrial', 'finance', 'investing', 'all'
        --df= dataframe
    """
    from numpy import median

    def gtrends_file(cik, category):
        gtrends_dir = os.path.join(os.path.expanduser('~'), \
                        'Dropbox', 'gtrends-beta', 'cik-ipo', category)
        return os.path.join(gtrends_dir, str(cik)+'.csv')

    def get_date_index(ref_date, iot_triple):
        for triple in iot_triple:
            index, date, iot = triple
            if arrow.get(date) < arrow.get(ref_date):
                continue
            else:
                return index
        return len(iot_triple) - 1

    for cik in df.index:
        print('\rCrunching interest-over-time <{}>: {} {}'.format(
                category, cik, firmname(cik)), ' '*40, end='\r')

        firm = df.loc[cik]
        roadshow_start = arrow.get(firm['date_1st_pricing'])
        # Start measuring IoT arbitrary 60 days before roadshow.
        start_date = roadshow_start
        s1_date = arrow.get(FINALJSON[cik]['Filing'][-1][2], 'M/D/YYYY')


        if firm['days_to_first_price_update'] < firm['days_to_final_price_revision']:
            end_date = roadshow_start.replace(days=firm['days_to_first_price_update'])
        elif firm['days_to_final_price_revision'] == firm['days_to_final_price_revision']:
            end_date = roadshow_start.replace(days=firm['days_to_final_price_revision'])
        else:
            # No price revision whatsoever
            end_date = arrow.get(firm.date_trading)

        iot_data = pd.read_csv(gtrends_file(cik=cik, category=category))
        iot_triple = list(iot_data[iot_data.columns[:2]].itertuples())
        # get first 2 columns
        iot = [x[2] for x in iot_triple]
        try:
            S = start_index = get_date_index(start_date, iot_triple)
            E = end_index = get_date_index(end_date, iot_triple)
            M = s1_index = get_date_index(s1_date, iot_triple)
            # median IoT in previous 60 days before s-1 filing
            median_iot = median(iot[M-70:M-14])
            # -10 weeks to -2 weeks before S-1 filing to calculate median IoT
            # Cumulative search interest after roadshow begins:
            CSI = sum(iot[S-30:E])
            # Cumulative abnormal search interest:
            CASI = sum(i - median_iot for i in iot[E-30:E])
        except (ValueError, IndexError) as e:
            print(start_date, 'not in', iot_triple[0][0], "-", iot_triple[-1][0])
            median_iot = 0
            CSI  = 0
            CASI = 0

        if median_iot != median_iot:
            median_iot, CSI, CASI = 0, 0, 0

        if len(iot_data.columns)==4:
            if iot_data.columns[1] not in iot_data.columns[3]:
                print("{}: {} =>\n\tSearched:\t\t'{}'\n\tGtrends_Entity_Name:\t'{}'".format(
                        cik, firmname(cik), iot_data.columns[1], iot_data.columns[3]))

        if 'IoT_entity_type' in df.keys():
            if iot_data.columns[2] != 'Search term' and df.loc[cik, 'IoT_entity_type'] == 'Search term':
                entity_type = iot_data.columns[2]
            else:
                entity_type = df.loc[cik, 'IoT_entity_type']

        df.loc[cik, 'gtrends_name'] = iot_data.columns[1]
        df.loc[cik, 'IoT_entity_type'] = entity_type
        df.loc[cik, 'IoT_CASI_{}'.format(category)] = CASI
        df.loc[cik, 'IoT_CSI_{}'.format(category)] = CSI
        df.loc[cik, 'IoT_median_{}'.format(category)] = median_iot

    entity_types = Counter(df['IoT_entity_type'])
    return entity_types


def weighted_iot(df, weighting='value_weighted'):

    all_iot = sum(1 for x in df["IoT_CASI_all"] if x != 0)
    finance = sum(1 for x in df["IoT_CASI_finance"] if x != 0)
    # busnews = sum(1 for x in df["IoT_CASI_business-industrial"] if x != 0)
    # invest  = sum(1 for x in df["IoT_CASI_investing"] if x != 0)
    # total   = all_iot + busnews + finance + invest
    total = all_iot + finance

    if weighting=='value_weighted':
        wa = weight_all_iot = all_iot/total
        wf = weight_finance = finance/total
        # wb = weight_busnews = busnews/total
        wf = weight_finance = finance/total
        # wi = weight_invest  = invest/total
    else:
        # wa, wb, wf, wi = [1/4, 1/4, 1/4, 1/4]
        wa, wf = [1/2, 1/2]

    df['IoT_median_all_finance'] = wa*df['IoT_median_all'] + wf*df['IoT_median_finance']
    df['IoT_CASI_all_finance'] = wa*df['IoT_CASI_all'] + wf*df['IoT_CASI_finance']
    df['IoT_CSI_all_finance'] = wa*df['IoT_CSI_all'] + wf*df['IoT_CSI_finance']

    # df['IoT_median_all_finance'] = wa*df['IoT_median_all'] + wb*df['IoT_median_business-industrial'] + wf*df['IoT_median_finance'] + wi*df['IoT_median_investing']

    # df['IoT_CASI_all_finance'] = wa*df['IoT_CASI_all'] + wb*df['IoT_CASI_business-industrial'] + wf*df['IoT_CASI_finance'] + wi*df['IoT_CASI_investing']

    # df['IoT_CSI_all_finance'] = wa*df['IoT_CSI_all'] + wb*df['IoT_CSI_business-industrial'] + wf*df['IoT_CSI_finance'] + wi*df['IoT_CSI_investing']

    df['ln_CASI_all_finance'] = np.log(df['IoT_CASI_all_finance'] + 1)
    df['ln_CSI_all_finance'] = np.log(df['IoT_CSI_all_finance'] + 1)

    return df






def plot_iot(cik, category=''):
    "plots Google Trends Interest-over-time (IoT)"


    def gtrends_file(cik, category):
        gtrends_dir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'gtrends-beta', 'cik-ipo', category)
        return os.path.join(gtrends_dir, str(cik)+'.csv')

    # import seaborn as sb

    # def rgb_to_hex(rgb):
    #     rgb = map(lambda x: int(max(0, min(x, 255)) * 255), rgb)
    #     return "#{0:02x}{1:02x}{2:02x}".format(*rgb)


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



def process_IOT_variables():
    entities1 = cumulative_iot(df, 'all')
    entities2 = cumulative_iot(df, 'finance')
    entities3 = cumulative_iot(df, 'business-industrial')
    # entities4 = cumulative_iot(df, 'investing')

    df = weighted_iot(df, weighting='equal')

    df['ln_CASI_all'] = np.log(df['IoT_CASI_all'] + 1)
    df['ln_CSI_all'] = np.log(df['IoT_CSI_all'] + 1)

    df.to_csv("df.csv")










if __name__=='__main__':

    FINALJSON = json.loads(open('final_json.txt').read())
    ciks = sorted(FINALJSON.keys())

    company     = pd.DataFrame([FINALJSON[cik]['Company Overview']  for cik in ciks], ciks)
    financials  = pd.DataFrame([FINALJSON[cik]['Financials']        for cik in ciks], ciks)
    experts     = pd.DataFrame([FINALJSON[cik]['Experts']           for cik in ciks], ciks)
    metadata    = pd.DataFrame([FINALJSON[cik]['Metadata']          for cik in ciks], ciks)
    filings     = pd.DataFrame([FINALJSON[cik]['Filing']            for cik in ciks], ciks)
    open_prices = pd.DataFrame([FINALJSON[cik]['Opening Prices']    for cik in ciks], ciks)

    df = pd.read_csv('df.csv', dtype={'cik':object})
    df.set_index('cik', inplace=True)
    sample = df[~df.size_of_first_price_update.isnull()]


    cik = '1439404' # Zynga         # 8.6
    cik = '1418091' # Twitter       # 7.4
    cik = '1271024' # LinkedIn      # 9.6
    cik = '1500435' # GoPro         # 8.1
    cik = '1318605' # Tesla Motors  # 8
    cik = '1326801' # Facebook      # 8.65
    cik = '1564902' # SeaWorld      # 9.54
    cikfb = '1326801' # Facebook

    ciks = ['1439404', '1418091', '1271024', '1500435', '1318605', '1594109', '1326801', '1564902']
    searchterms = {k:v for k,v in df[(df.IoT_entity_type=='Search term')]['gtrends_name'].items()}


