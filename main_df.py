

import glob, json, os, re
import pandas as pd
import numpy as np
import requests
import arrow
import matplotlib.pyplot as plt
import seaborn as sb

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

        # '1467858' -> GM

    def create_dataframe(category, ciks=sorted(FINALJSON.keys())):
        return pd.DataFrame([FINALJSON[cik][category] for cik in ciks], ciks)



# None of these functions return a DF, they modify existing df.
def df_filings(FINALJSON=FINALJSON):
    """Takes FINALJSON dataframes, gets price revision stats"""

    def first_price_update(dates, prices, pranges):
        for i, p in enumerate(prices[:-1]):
            if p == prices[0]:
                continue
            diff_dates  = (dates[i] - dates[0]).days
            diff_prices = (prices[i] - prices[0])
            diff_prange = (pranges[i] - pranges[0])
            percent_price_change = diff_prices / prices[0]
            return diff_dates, diff_prices, percent_price_change, diff_prange
        return [None, None, None, None]

    def num_price_updates_up(prices):
        return sum(1 for f,s in zip(prices, prices[1:]) if f<s)

    def num_price_updates_down(prices):
        return sum(1 for f,s in zip(prices, prices[1:]) if f>s)

    def num_price_updates(prices):
        return sum(1 for f,s in zip(prices, prices[1:]) if f!=s)

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


    ciks = sorted(FINALJSON.keys())
    ###########
    print('Creating a new dataframe...')
    float_columns = ['days_to_first_price_update',
                    'days_to_final_price_revision',
                    'days_from_priced_to_listing',
                    'size_of_first_price_update',
                    'size_of_final_price_revision',
                    'percent_first_price_update',
                    'percent_final_price_revision',
                    'prange_change_first_price_update',
                    'number_of_price_updates']
    df = pd.DataFrame([[0]*len(float_columns)]*len(ciks), ciks, columns=float_columns)
    ##########


    company     = create_dataframe('Company Overview')
    financials  = create_dataframe('Financials')
    experts     = create_dataframe('Experts')
    metadata    = create_dataframe('Metadata')
    filings     = create_dataframe('Filing')
    open_prices = create_dataframe('Opening Prices')


    RW, MISSING, ABNORMAL = [], [], set()
    badciks = {     # long durations
        '1481582': 9, # Ryerson Holding
        '1087294': 12 # Cumberland Pharma
        }

    for cik, values in filings.iterrows():
        iprint("Parsing price data for {}:{}".format(cik, firmname(cik)))
        if any(l[1] == 'RW' for l in values if l):
            RW.append(cik)

        values = [v for v in values if v]
        if cik in badciks:
            values = values[:badciks[cik]]
        values = [v for v in values if is_cash(v[-1][0]) and as_cash(v[-1][0]) >= 3]
        if not values:
            MISSING.append(cik)
            print("{}: {} no filings above 3$".format(cik, firmname(cik)))
            continue

        pranges = [prange(v[-1]) for v in values]
        prices = [midpoint(v[-1]) for v in values]
        dates  = [aget(v[2]).date() for v in values]
        if len(set(dates)) < len(dates):
            # tie-break same date filings
            dates, prices, pranges = (list(reversed(x)) for x in [dates, prices, pranges])
        else:
            dates, prices, pranges = zip(*sorted(zip(dates, prices, pranges)))

        trade_date = arrow.get(open_prices.loc[cik, 'Date']).date()
        B424_date = aget(values[0][2]).date()
        priced_date = aget(company.loc[cik, 'Status']).date()
        s1_date = aget([x for x in filings.loc[cik] if x][-1][2]).date()

        if (trade_date - priced_date).days > 90:
            # trading_date - priced_date > 1: => Firm floats with delay
            if (trade_date - B424_date).days > 90:
                # 424B_date - priced_date: delayed final prospectus
                print("{}: {} =>\n\tTrade date: {}\n\t424B Date: {}\n\tPriced: {}".format(cik, firmname(cik), trade_date, B424_date, priced_date))
                ABNORMAL |= {cik}

        # NASDAQ 'priced_date' can be early, get true final revision price date
        if priced_date < B424_date < trade_date:
            final_rev_date = B424_date
        elif B424_date < priced_date  < trade_date:
            final_rev_date = priced_date
        elif priced_date == B424_date < trade_date:
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
            number_of_price_updates = 0
            number_of_price_updates_up = 0
            number_of_price_updates_down = 0
            days_to_final_price_revision = None
            size_of_final_price_revision = None
            percent_final_price_revision = None
            print("{}: {} has 1 filing".format(cik, firmname(cik)))

        elif len(prices) > 1:
            if dates[1] == final_rev_date:
                days_to_first_price_update = None
                size_of_first_price_update = None
                percent_first_price_update = None
                # prange_change_first_price_update =
            else:
                days_to_first_price_update, \
                size_of_first_price_update, \
                percent_first_price_update, \
                prange_change_first_price_update = first_price_update(dates,prices,pranges)

                if size_of_first_price_update:
                    if size_of_first_price_update > 10:
                        ABNORMAL |= {cik}
                        print("{}: {} has large price updates".format(cik, firmname(cik)))

            number_of_price_updates = num_price_updates(prices)
            number_of_price_updates_up = num_price_updates_up(prices)
            number_of_price_updates_down = num_price_updates_down(prices)
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

        if percent_first_price_update:
            pct_first_price_change = percent_first_price_update
        else:
            pct_first_price_change = percent_final_price_revision

        if days_to_first_price_update:
            days_to_first_price_change = days_to_first_price_update
        else:
            days_to_first_price_change = days_to_final_price_revision

        df.loc[cik, 'pct_first_price_change'] = pct_first_price_change
        df.loc[cik, 'days_to_first_price_change'] = days_to_first_price_change
        df.loc[cik, 'days_from_priced_to_listing'] = (trade_date - dates[0]).days
        df.loc[cik, 'offer_in_filing_price_range'] = offer_in_filing_price_range
        df.loc[cik, 'prange_change_first_price_update'] = prange_change_first_price_update
        df.loc[cik, 'number_of_price_updates'] = number_of_price_updates
        df.loc[cik, 'number_of_price_updates_up'] = number_of_price_updates_up
        df.loc[cik, 'number_of_price_updates_down'] = number_of_price_updates_down
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

    ###############
    df['cik'] = df.index
    df.to_csv("df_temp.csv", dtype={'cik':object})

    return df



if '__control_variables___':
    def order_df(df):
        "Corrects dtypes and reorders keys in dataframe"

        ipo_cycle_keys = [k for k in df.keys() if k[1].isdigit()]
        iot_keys = [k for k in df if re.search(r'IoT_\d', k)]
        cols = [
            'days_to_first_price_update',
            'days_to_final_price_revision',
            'days_from_priced_to_listing',
            'days_from_s1_to_listing',
            'size_of_first_price_update',
            'size_of_final_price_revision',
            'percent_first_price_update',
            'percent_final_price_revision',
            'pct_first_price_change',
            'days_to_first_price_change',
            'prange_change_first_price_update', 'prange_change_plus',
            'delay_in_price_update',
            'number_of_price_updates',
            'number_of_price_updates_up', 'number_of_price_updates_down',
            'offer_in_filing_price_range',
            'Coname', 'Year',
            'date_1st_pricing',  'date_last_pricing',
            'date_s1_filing', 'date_424b', 'date_trading',
            'Date', 'Offer Price', 'Open', 'Close', 'High', 'Low', 'Volume',
            'open_return', 'close_return',
            'SIC', 'FF49_industry', 'BAA_yield_changes',
            'confidential_IPO', 'foreign', 'exchange', 'amends', 'VC',
            'underwriter_rank_single', 'underwriter_rank_avg', 'underwriter_rank_med',
            'underwriter_num_leads', 'underwriter_collective_rank', 'underwriter_tier',
            'share_overhang', 'shares_offered', 'total_dual_class_shares',
            'proceeds', 'market_cap', 'liab_over_assets',
            'price_to_earnings', 'price_to_sales', 'EPS', 'sales',
            'priceupdate_down', 'priceupdate_up'
            ] + ipo_cycle_keys + iot_keys

        missed_cols = sorted(set(df.keys()) - set(cols))


        col_float = [
            'days_to_final_price_revision', 'days_from_priced_to_listing',
            'days_from_s1_to_listing', 'days_to_first_price_change',
            'days_to_first_price_update', 'delay_in_price_update',
            'size_of_first_price_update', 'size_of_final_price_revision',
            'percent_first_price_update', 'percent_final_price_revision',
            'pct_first_price_change', 'prange_change_first_price_update',
            'Offer Price', 'Open', 'Close', 'High', 'Low', 'Volume',
            'open_return', 'close_return',  'BAA_yield_changes',
            'underwriter_rank_single', 'underwriter_rank_avg', 'underwriter_rank_med',
            'underwriter_collective_rank', 'share_overhang', 'shares_offered',
            'proceeds', 'market_cap', 'liab_over_assets',
            'price_to_earnings', 'price_to_sales', 'EPS', 'sales',
            'priceupdate_down', 'priceupdate_up',
            'total_dual_class_shares'
            ] + ipo_cycle_keys + iot_keys
        col_int = ['Year', 'confidential_IPO', 'underwriter_num_leads',
            'foreign', 'VC', 'number_of_price_updates',
            'number_of_price_updates_up', 'number_of_price_updates_down']
        col_obj = ['offer_in_filing_price_range', 'Coname', 'date_1st_pricing', 'date_last_pricing', 'date_424b', 'date_trading', 'date_s1_filing', 'Date', 'SIC', 'FF49_industry', 'underwriter_tier', 'exchange', 'amends',
            ]

        df[col_float] = df[col_float].astype(float)
        df[col_int] = df[col_int].astype(int)
        df[col_obj] = df[col_obj].astype(object)

        if not isinstance(df['SIC'].iloc[0], str):
            df["SIC"] = ['0'+str(i) if i<1000 else str(i) for i in df["SIC"]]

        return df[cols + missed_cols]


    def underwriter_ranks(df):
        """Gets the Ritter/Carter-Manaster rank for the 'Lead Underwriters' in the supplied dataframe
        dataframe musthave keys: lead_underwriter and s1_date
        """

        from functools import partial
        from arrow     import Arrow
        from fuzzywuzzy import process
        uw_rank = pd.read_csv(BASEDIR + "/data/uw_rank.csv", encoding="latin-1")
        uw_rank = uw_rank.dropna()
        uw_rank = uw_rank.set_index("Underwriter")
        na_uw = {'No Underwriter', 'Self-underwritten', ' -- '}

        def is_same_alphabet(a_reference, a):
            return a.lower().startswith(a_reference)

        def lookup_underwriter(uw):
            "Matches UW with UR from carter manaster ranks by fuzzystring matching."
            is_same_alpha = partial(is_same_alphabet, uw[0].lower())
            uw_with_same_alpha = sorted(filter(is_same_alpha, uw_rank.index))
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

        def uw_tier(uw_rank):
            # top_tier = {'Citigroup', 'Credit Suisse', 'Goldman Sachs', 'Merrill Lynch'}
            if uw_rank > 8.5:
                return "8.5+"
            elif uw_rank > 7:
                return "7+"
            elif uw_rank >= 0:
                return "0+"
            elif uw_rank < 0:
                return "-1"

        # Fuzzy match to Carter-Manaster underwriter names
        for cik, uw_list in experts['Lead Underwriter'].items():
            # Filter no underwriter
            if uw_list != uw_list:
                df.loc[cik, 'underwriter_rank_single'] = -1
                df.loc[cik, 'underwriter_rank_avg'] = -1
                df.loc[cik, 'underwriter_rank_med'] = -1
                df.loc[cik, 'underwriter_num_leads'] = 0
                df.loc[cik, 'underwriter_collective_rank'] = -1
                continue
            if all(uw in na_uw for uw in uw_list):
                df.loc[cik, 'underwriter_rank_single'] = -1
                df.loc[cik, 'underwriter_rank_avg'] = -1
                df.loc[cik, 'underwriter_rank_med'] = -1
                df.loc[cik, 'underwriter_num_leads'] = 0
                df.loc[cik, 'underwriter_collective_rank'] = -1
                continue

            print("Getting underwriter rank for {}: {}".format(cik, firmname(cik)))
            uw_ranks = [get_uw_rank(uw,cik) for uw in uw_list]
            # #Average underwriter rank
            CM_rank_single  = max(uw_ranks)
            CM_rank_average = round(sum(uw_ranks)/len(uw_list), 1)
            CM_rank_median  = median(uw_ranks)
            df.loc[cik, 'underwriter_rank_single'] = CM_rank_single
            df.loc[cik, 'underwriter_rank_avg'] = CM_rank_average
            df.loc[cik, 'underwriter_rank_med'] = CM_rank_average
            df.loc[cik, 'underwriter_num_leads'] = len(uw_list)
            df.loc[cik, 'underwriter_collective_rank'] = round(sum(uw_ranks), 1)
            # df.loc[cik, 'underwriter_tier'] = uw_tier(CM_rank_average)

        df['underwriter_tier'] = [uw_tier(r) for r in df['underwriter_rank_avg']]
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


    def ipo_cycle_variables(df, lag=3, date_key='date_1st_pricing', weighting='value'):
        """Gets FF49 industry returns, and IPO deal volume
            for each firm in <df> for the <lag> months leading up to an IPO.
        """

        import pandas.io.data as web
        import re, arrow, io, requests, datetime
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
                if not date:
                    df_returns += [None]
                    continue

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

            start = sorted([x for x in filing_dates if x])[0].date()
            end = sorted([x for x in filing_dates if x])[-1].date()
            print("Grabbing Moody's BAA corporate yield bond data from FRED...")
            BAA = web.DataReader('BAA', 'fred', start, end)
            BAA.index = [arrow.get(s) for s in BAA.index]

            BAA_yield_changes = []
            for d in filing_dates:
                if not d:
                    BAA_yield_changes += [None]
                    continue

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

            ipo_cycles = pd.read_csv("/Users/peitalin/Data/IPO/NASDAQ/data/master-2005-2014.idx",
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


    def get_financials(df):
        "Populate df with financials metrics: liabilities/assets, EPS, log_proceeds"


        def midpoint(iterable):
            assert len(list(iterable)) <= 2
            return sum(list(iterable)) / len(iterable)

        def as_int(string):
            return int(string.replace(',',''))

        filings = pd.DataFrame([FINALJSON[cik]['Filing'] for cik in ciks], ciks)

        for cik in df.index:
            print("Getting financial ratios for {}: {}".format(cik, firmname(cik)), end=" "*20+'\r')
            FJ = FINALJSON[cik]

            priceranges = [f[-1] for f in filings.loc[cik] if f and f[-1]!=['NA']]
            priceranges = [list(map(as_cash, pr)) for pr in priceranges if as_cash(pr[0])]
            try:
                midpoint_initial_pr = [midpoint(pr) for pr in priceranges][-1]
                proceeds = midpoint_initial_pr * as_int(FJ['Company Overview']['Shares Offered'])
            except IndexError:
                proceeds = None

            net_income = as_cash(FJ['Financials']['Net Income'])
            revenue = as_cash(FJ['Financials']['Revenue'])
            liabil = as_cash(FJ['Financials']['Total Liabilities'])
            assets = as_cash(FJ['Financials']['Total Assets'])
            num_shares = df.loc[cik, 'total_dual_class_shares']

            if num_shares != ' -- ' and num_shares != None:
                num_shares = float(num_shares)
                market_cap = num_shares * midpoint_initial_pr
            else:
                num_shares = None
                market_cap = None

            df.loc[cik, 'market_cap']   = market_cap
            df.loc[cik, 'proceeds'] = proceeds if proceeds else 0
            df.loc[cik, 'liab_over_assets'] = liabil/assets if liabil and assets else 0
            df.loc[cik, 'price_to_earnings'] = market_cap/net_income if net_income and market_cap else 0
            df.loc[cik, 'sales'] = revenue if revenue else 0
            df.loc[cik, 'price_to_sales'] = market_cap/revenue if revenue and market_cap else 0
            df.loc[cik, 'EPS'] = net_income/num_shares if net_income and num_shares else 0
            float_cols = ['liab_over_assets', 'price_to_earnings', 'price_to_sales', 'EPS']
            df[float_cols] = df[float_cols].astype(float)

        return df


    def misc_variables(df):

        iprint("Setting misc variables...")
        df['percent_first_price_update'] = [x if x==x else 0 for x in df['percent_first_price_update']]

        df['priceupdate_up'] = [x if x>0 else 0 for x in df['percent_first_price_update']]
        df['priceupdate_down'] = [x if x<0 else 0 for x in df['percent_first_price_update']]

        df['pct_final_revision_up'] = [x if x>0 else 0 for x in df['percent_final_price_revision']]
        df['pct_final_revision_down'] = [x if x<0 else 0 for x in df['percent_final_price_revision']]


        df['prange_change_plus'] = [abs(x) if x<0 else 0 for x in df['prange_change_first_price_update']]

        df['days_from_s1_to_listing'] = [(aget(d2)-aget(d1)).days for d1,d2 in zip(df['date_s1_filing'], df['date_424b'])]


        def delay_in_price_update(cik):
            time_to_price_update = df.loc[cik, 'days_to_first_price_change']
            roadshow_duration = df.loc[cik, 'days_from_priced_to_listing']
            if np.isnan(time_to_price_update/roadshow_duration):
                return 1 # 1st pricing on the day before listing
            else:
                return time_to_price_update/roadshow_duration

        def merge_exchange(cik):
            exchange = company.loc[cik, 'Exchange']
            if exchange in ['NASDAQ National Market', 'NASDAQ Global Market']:
                return 'NASDAQ Global Market'
            elif 'Smallcap' in exchange:
                return 'NASDAQ Smallcap Market'
            else:
                return exchange

        def amends_category(cik):
            p_update = df.loc[cik, 'size_of_first_price_update']
            if p_update == 0:
                return "Same"
            elif p_update < 0:
                return "Down"
            elif p_update > 0:
                return "Up"
            elif p_update != p_update:
                return "None"

        df['delay_in_price_update'] = list(map(delay_in_price_update, df.index))
        df['exchange'] = list(map(merge_exchange, df.index))
        df['amends'] = list(map(amends_category, df.index))

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
    df['Offer_Price'] = [as_cash(s) for s in company.sort_index()['Share Price']]
    df['open_return'] = (df['Open'] - df['Offer Price']) / df['Offer Price']
    df['close_return'] = (df['Close'] - df['Offer Price']) / df['Offer Price']
    df["SIC"] = metadata["SIC"]
    df['Coname'] = company['Company Name']


    ########
    df = underwriter_ranks(df)
    df = match_sic_indust_FF(df)
    df = dual_class_shares(df)
    df = share_overhang(df)
    df = confidential_IPO(df)
    df = ipo_cycle_variables(df)
    df = get_financials(df)
    df = misc_variables(df)
    df = order_df(df)
    # df.to_csv("df.csv", dtype={"cik": object, 'SIC':object, 'Year':object})


def count_S1A_during_roadshow():

    def tailcount(l):
        if len(l) < 1:
            return 0

        tail = l[-1]
        i=0
        while tail == l.pop():
            i+=1
            if len(l) < 1:
                return i
        return i

    num_roadshow_S1A = [tailcount([as_cash(x[-1][0]) for x in FINALJSON[cik]['Filing']
                        if as_cash(x[-1][0])]) for cik in ciks]
    df['num_roadshow_S1A'] = num_roadshow_S1A
    df.to_csv("df.csv", dtype={"cik": object, 'SIC':object, 'Year':object})



if '__Attention_variables__':

    def attention_measures(df, category, event='final', makedir=True):
        """Calculates ASI (abnormal search interest) and CASI (cumulative abnormal search interest)
            ASI = IoT - IoT_{n}day_median ({n}=15, 30, or 60 days back)
        Args:
            --category: 'business_industrial', 'finance', 'all'
            --df: dataframe or list of firm cik identifiers
            --event: ['final', '1st_update', 'listing']
        """

        from pandas.stats.moments import rolling_median, rolling_sum
        QDIR = os.path.join(os.path.expanduser('~'), 'Dropbox', 'gtrends-beta', 'cik-ipo', 'query_counts')
        QWEIGHTS = {'missing':0, 'weekly': 0.5, 'daily': 1}
        CATEGORYID = {'all': '0', 'finance': '0-7', 'business_industrial': '0-12'}

        ciks = tuple(df.index) if isinstance(df, pd.core.frame.DataFrame) else df


        def build_qcount_dict(category):
            qdict = {}
            for cik in ciks:
                iprint('Building qcount list: %s' % cik)
                qfile = os.path.join(QDIR, CATEGORYID[category], cik+'.csv')
                with open(qfile) as f:
                    qcounts = [x.strip().split(',')[-1] for x in f.readlines()[1:]]
                qdict.update({cik:qcounts})
            return qdict

        QDICT = build_qcount_dict(category)

        def gtrends_file(cik, category):
            gtrends_dir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'gtrends-beta', 'cik-ipo', category)
            # gtrends_dir = os.path.join(BASEDIR, 'cik-ipo', category)
            return os.path.join(gtrends_dir, str(cik)+'.csv')

        def make_dir(cik, category):
            fdir = os.path.join(BASEDIR, 'IoT', category)
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            iot.to_csv(os.path.join(fdir, 'IoT_'+cik+'.csv'))

        def get_end_date(cik, event='final'):
            "Args: event: ['final', '1st_update', 'listing'] "

            firm = df.loc[cik]
            start_date = aget(firm['date_1st_pricing'])
            trade_date = aget(firm['date_trading'])

            if event=='final':
                if firm['days_to_final_price_revision'] and not np.isnan(firm['days_to_final_price_revision']):
                    end_date = start_date.replace(days=firm['days_to_final_price_revision'])
                else:
                    end_date = trade_date

            elif event=='listing':
                end_date = trade_date.replace(days=1)

            elif event=='postlisting':
                end_date = trade_date.replace(days=15)

            else:
                if firm['days_to_first_price_update'] < firm['days_to_final_price_revision']:
                    end_date = start_date.replace(days=firm['days_to_first_price_update'])
                elif not np.isnan(firm['days_to_final_price_revision']):
                    end_date = start_date.replace(days=firm['days_to_final_price_revision'])
                else:
                    end_date = aget(firm['date_trading'])

            return end_date.date()

        def get_entity_type(iot_data , df):
            if 'IoT_entity_type' in df.keys():
                if iot_data.columns[1] != 'Search term' and df.loc[cik, 'IoT_entity_type'] == 'Search term':
                    entity_type = iot_data.columns[1]
                else:
                    entity_type = df.loc[cik, 'IoT_entity_type']
            return entity_type

        def lambda_param(cik, category):
            "Calculates scaling factor depending on whether iot is daily or weekly or none."
            qcounts = QDICT[cik]
            return sum(QWEIGHTS[q] for q in qcounts) / len(qcounts)

        def box_cox(CASI, lambda_param=0):
            "Returns the John & Draper (1980) modulus transformation"
            if lambda_param == 0:
                return sign(CASI) * log(1 + abs(CASI))
            else:
                return sign(CASI) * ((1 + abs(CASI))**lambda_param - 1) / lambda_param

        def rolling_attention(iot, window):
            "callback function for ASI calculations"
            w = window
            firm = iot.columns[0]
            iot['%sday_median' % w] = rolling_median(iot[firm], w)
            iot['%sday_ASI' % w] = log((1 + iot[firm]) / (1 + iot['%sday_median' % w]))
            iot['%sday_CASI' % w] = rolling_sum(iot['%sday_ASI' % w], w)
            return iot


        # ciks = ['1439404', '1418091', '1271024', '1500435', '1318605', '1594109', '1326801', '1564902']

        for i, cik in enumerate(ciks):
            iprint('Computing interest-over-time <{}, {}>: {} {}'.format(
                    category, event, cik, firmname(cik)))

            iot_raw_data = pd.read_csv(gtrends_file(cik=cik, category=category),
                                    parse_dates=[0],
                                    date_parser=lambda d: aget(d).date(),
                                    index_col="Date")
            firm = iot_raw_data.columns[0]
            iot = iot_raw_data[iot_raw_data.columns[:1]]
            iot[firm] = box_cox(iot[firm], lambda_param=lambda_param(cik, category))

            iot = rolling_attention(iot, window=15)
            iot = rolling_attention(iot, window=30)
            # iot = rolling_attention(iot, window=60)

            try:
                end_date = get_end_date(cik, event=event)
                df.loc[cik, 'gtrends_name'] = firm
                df.loc[cik, 'IoT_entity_type'] = get_entity_type(iot_raw_data, df)

                df.loc[cik, 'IoT_15day_CASI_%s' % category] = iot['15day_CASI'].loc[end_date]
                df.loc[cik, 'IoT_30day_CASI_%s' % category] = iot['30day_CASI'].loc[end_date]
                # df.loc[cik, 'IoT_60day_CASI_%s' % category] = iot['60day_CASI'].loc[end_date]
            except:
                raise(KeyError("cik: {}".format(cik)))
            if makedir:
                make_dir(cik, category)

        return df


    def weighted_iot(df, window=15, weighting='equal'):

        w = window

        if weighting=='value_weighted':
            all_iot = sum(1 for x in df["IoT_{}day_CASI_all".format(w)] if x != 0)
            finance = sum(1 for x in df["IoT_{}day_CASI_finance".format(w)] if x != 0)
            bus_ind = sum(1 for x in df["IoT_{}day_CASI_business_industrial".format(w)] if x != 0)
            total = all_iot + finance + bus_ind
            wa = weight_all_iot = all_iot/total
            wf = weight_finance = finance/total
            wb = weight_bus_ind = bus_ind/total
        else:
            wa, wf, wb = [1/3, 1/3, 1/3]

        df['IoT_{}day_CASI_weighted_finance'.format(w)] = \
            wa * df['IoT_{}day_CASI_all'.format(w)] + \
            wf * df['IoT_{}day_CASI_finance'.format(w)] + \
            wb * df['IoT_{}day_CASI_business_industrial'.format(w)]
        return df


    def news_iot(df, window=15):

        def news_casi(cik, w=window):
            casi_fin = df.loc[cik, 'IoT_{}day_CASI_finance'.format(w)]
            casi_all = df.loc[cik, 'IoT_{}day_CASI_all'.format(w)]
            casi_bus = df.loc[cik, 'IoT_{}day_CASI_business_industrial'.format(w)]
            if casi_fin==0 and casi_bus==0:
                return casi_all
            elif casi_bus==0:
                return casi_fin
            elif casi_fin==0:
                return casi_bus
            else:
                return np.mean([casi_fin, casi_bus])

        df['IoT_{}day_CASI_news'.format(window)] = list(map(news_casi, df.index))
        return df


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
        colors = ['#0072b2', '#009e73', '#e58e20', '#cc79a7', '#f0e442', '#56b4e9']


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
        iot_data = iot_data[240:-360]

        firm = iot_data.columns[1]
        iot_data['Date'] = [arrow.get(d).date() for d in iot_data['Date']]
        # iot_melt = pd.melt(iot_data.icol([0,1]), id_vars=['Date'])
        # iot_melt.columns = ['Date', firm, 'Interest-over-time']
        # ax = iot_melt.groupby([firm, 'Date']).mean().unstack('Tesla Motors').plot()

        s1_date = arrow.get(df.loc[cik, 'date_s1_filing']).date()
        anno_index1 = iot_data[iot_data.Date == s1_date].index[0]

        roadshow_date = arrow.get(df.loc[cik, 'date_1st_pricing']).replace(days=-7).date()
        anno_index2 = iot_data[iot_data.Date == roadshow_date].index[0]

        date_listed = arrow.get(df.loc[cik, 'date_trading']).date()
        anno_index3 = iot_data[iot_data.Date == date_listed].index[0]


        fig, ax = plt.subplots(sharex=True, figsize=(15,5))
        ax.plot(iot_data["Date"], iot_data[firm], label='Search Interest: {} ({})'.format(firm, iot_data.columns[2]))
        ax.annotate('S-1 Filing',
                    (mdates.date2num(iot_data.Date[anno_index1]), iot_data[firm][anno_index1]),
                    xytext=(40, 40),
                    size=12,
                    color=colors[2],
                    textcoords='offset points',
                    arrowprops=dict(width=1.5, headwidth=5, shrink=0.1, color=colors[2]))

        ax.annotate('Roadshow Begins',
                    (mdates.date2num(iot_data.Date[anno_index2]), iot_data[firm][anno_index2]),
                    xytext=(-120, 40),
                    size=12,
                    color=colors[2],
                    textcoords='offset points',
                    arrowprops=dict(width=1.5, headwidth=5, shrink=0.1, color=colors[2]))

        ax.annotate('IPO Listing Date',
                    (mdates.date2num(iot_data.Date[anno_index3]), iot_data[firm][anno_index3]),
                    xytext=(-120, -50),
                    size=12,
                    color=colors[2],
                    textcoords='offset points',
                    arrowprops=dict(width=1.5, headwidth=5, shrink=0.1, color=colors[2]))

        plt.title("Interest-over-time for {} - ({})".format(firm, category))
        plt.ylabel("Search Interest")
        plt.legend()
        plt.show()


    def abnormal_svi(df, window=15, category='all'):

        def get_mid_date(cik, event='final'):
            "Args: event: ['final', '1st_update', 'listing'] "
            firm = df.loc[cik]
            start_date = aget(firm['date_1st_pricing'])
            trade_date = aget(firm['date_trading'])
            if event=='final':
                if not np.isnan(firm['days_to_final_price_revision']):
                    end_date = start_date.replace(days=firm['days_to_final_price_revision'])
                else:
                    end_date = trade_date
            elif event!='listing':
                if firm['days_to_first_price_update'] < firm['days_to_final_price_revision']:
                    end_date = start_date.replace(days=firm['days_to_first_price_update'])
                elif not np.isnan(firm['days_to_final_price_revision']):
                    end_date = start_date.replace(days=firm['days_to_final_price_revision'])
                else:
                    end_date = aget(firm['date_trading'])
            return end_date

        if os.path.exists("IoT/ASVI_{}day_{}.csv".format(window, category)):
            ASVI = pd.read_csv("IoT/ASVI_{}day_{}.csv".format(window, category), dtype={'cik': object}).set_index('cik')
            return ASVI

        w = window
        columns = ['t-{}'.format(d) for d in range(w+1)][::-1] + ['t+{}'.format(d) for d in range(1, w+1)]
        ASVI = []

        for cik in df.index:
            iprint("Getting ASVI => {}:{}".format(cik, firmname(cik)))
            fdir = os.path.join(BASEDIR, 'IoT/{}'.format(category), 'IoT_{}.csv'.format(cik))
            iot = pd.read_csv(fdir, parse_dates=[0], date_parser=aget)
            iot.set_index("Date", inplace=1)
            mid_date = get_mid_date(cik, event='final')
            drange = arrow.Arrow.range('day', mid_date.replace(days=-w), mid_date.replace(days=w))
            ASVI.append(list(iot.loc[drange, '{}day_CASI'.format(w)]))

        ASVI = pd.DataFrame(ASVI, index=df.index, columns=columns)
        ASVI.to_csv('IoT/ASVI_{}day_{}.csv'.format(window, category), dtype={'cik':object})
        return ASVI

        """
        if category='weighted_finance':
            ASVI15_ = (abnormal_svi(df, window=15, category='all') +
                      abnormal_svi(df, window=15, category='business_industrial') +
                      abnormal_svi(df, window=15, category='finance')) / 3
            ASVI30_ = (abnormal_svi(df, window=30, category='all') +
                      abnormal_svi(df, window=30, category='business_industrial') +
                      abnormal_svi(df, window=30, category='finance')) / 3
            ASVI60_ = (abnormal_svi(df, window=60, category='all') +
                      abnormal_svi(df, window=60, category='business_industrial') +
                      abnormal_svi(df, window=60, category='finance')) / 3
        """



def process_IOT_variables():

    entity_types = Counter(df['IoT_entity_type'])

    ### Attention until final price revision
    df = attention_measures(df, 'all', event='final')
    df = attention_measures(df, 'finance', event='final')
    df = attention_measures(df, 'business_industrial', event='final')

    df = weighted_iot(df, window=15, weighting='equal')
    df = weighted_iot(df, window=30, weighting='equal')

    df = news_iot(df, window=15)
    df = news_iot(df, window=30)

    df.to_csv("df.csv", dtype={"cik": object, 'SIC':object, 'Year':object})


    ### Attention until 1st price update
    dfu = attention_measures(dfu, 'all', event='1st_update')
    dfu = attention_measures(dfu, 'finance', event='1st_update')
    dfu = attention_measures(dfu, 'business_industrial', event='1st_update')

    dfu = weighted_iot(dfu, window=15, weighting='equal')
    dfu = weighted_iot(dfu, window=30, weighting='equal')
    dfu = news_iot(dfu, window=15)
    dfu = news_iot(dfu, window=30)

    dfu.to_csv("df_update.csv", dtype={"cik": object, 'SIC':object, 'Year':object})



    ### Attention post-listing, in 2 weeks after listing.
    dfl = df
    dfl = attention_measures(dfl, 'all', event='postlisting')
    dfl = attention_measures(dfl, 'finance', event='postlisting')
    dfl = attention_measures(dfl, 'business_industrial', event='postlisting')

    dfl = weighted_iot(dfl, window=15, weighting='equal')
    dfl = weighted_iot(dfl, window=30, weighting='equal')

    dfl = news_iot(dfl, window=15)
    dfl = news_iot(dfl, window=30)

    dfl.to_csv("dfl.csv", dtype={"cik": object, 'SIC':object, 'Year':object})


def uw_syndicate(experts):

    experts['Underwriter'] = [[] if type(x)!=list else x for x in experts['Underwriter']]
    experts['Lead Underwriter'] = [[] if type(x)!=list else x for x in experts['Lead Underwriter']]
    df['underwriter_syndicate_size'] = [0] * len(df)

    syndicate = []
    for cik in df.index:
        print(cik)
        syn_size = len(set(experts.ix[cik]['Lead Underwriter'] +  experts.ix[cik]['Underwriter']))
        syndicate.append(syn_size)

    df['underwriter_syndicate_size'] = syndicate

    df.to_csv("df.csv", dtype={"cik": object, 'SIC':object, 'Year':object})



def original_prange(ciks):

    pranges = []
    prange_pct_changes = []
    for cik in ciks:
        try:
            prange01 = [x[-1] for x in FINALJSON[cik]['Filing'] if len(x[-1]) > 1]
            prange01 = [x for x in prange01 if '$' not in x]
            prange0 = [as_cash(x) for x in prange01[-1]]
            for prange in prange01[::-1]:
                prange1 = [as_cash(x) for x in prange]
                if prange1 != prange0:
                    print("{} -> {}".format(prange0, prange1))
                    prange_pct_change = ((max(prange1) - min(prange1))/(max(prange0) - min(prange0)) - 1)* 100
                    break
            else:
                prange1 = None
                prange_pct_change = -100
        except:
            print("cik:{} has no prange0".format(cik))
            prange0 = ('NA')
            prange_pct_change = ('NA')
        pranges.append(prange0)
        prange_pct_changes.append(prange_pct_change)

    df['original_prange'] = pranges
    df['prange_change_pct'] = prange_pct_changes

    df.to_csv("df.csv", dtype={"cik": object, 'SIC':object, 'Year':object})



if __name__=='__main__':

    company     = create_dataframe('Company Overview')
    financials  = create_dataframe('Financials')
    experts     = create_dataframe('Experts')
    metadata    = create_dataframe('Metadata')
    filings     = create_dataframe('Filing')
    open_prices = create_dataframe('Opening Prices')

    # fulldf = pd.read_csv('full_df.csv', dtype={'cik': object})
    # fulldf.set_index('cik', inplace=True)

    df = pd.read_csv("df.csv", dtype={'cik':object, 'Year':object, 'SIC':object})
    dfu = pd.read_csv("df_update.csv", dtype={'cik':object, 'Year':object, 'SIC':object})
    df.set_index('cik', inplace=True)
    dfu.set_index('cik', inplace=True)


    ciks = sorted(FINALJSON.keys())
    cik = '1439404' # Zynga         # 8.6
    cik = '1418091' # Twitter       # 7.4
    cik = '1271024' # LinkedIn      # 9.6
    cik = '1500435' # GoPro         # 8.1
    cik = '1318605' # Tesla Motors  # 8
    cik = '1326801' # Facebook      # 8.65
    ciksea = '1564902' # SeaWorld      # 9.54
    cikfb = '1326801' # Facebook
    ciks1 = ['1439404', '1418091', '1271024', '1500435', '1318605', '1594109', '1326801', '1564902']
    iotkeys = ['gtrends_name', 'IoT_entity_type',
    'IoT_15day_CASI_all', 'IoT_30day_CASI_all', 'IoT_60day_CASI_all',
    'IoT_15day_CASI_finance', 'IoT_30day_CASI_finance', 'IoT_60day_CASI_finance',
    'IoT_15day_CASI_business_industrial','IoT_30day_CASI_business_industrial',
    'IoT_60day_CASI_business_industrial','IoT_15day_CASI_weighted_finance',
    'IoT_30day_CASI_weighted_finance', 'IoT_60day_CASI_weighted_finance']

    # len(df[(df.prange_change_pct != 'NA') & (df.delay_in_price_update<1)].prange_change_pct)

