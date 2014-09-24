

import glob, json, os, re
import pandas as pd
import numpy as np
import requests
import arrow
import matplotlib.pyplot as plt
import seaborn as sb

from collections        import Counter, OrderedDict
from pprint             import pprint
from concurrent.futures import ThreadPoolExecutor
from IPython            import embed

N = 10
FILE_PATH = 'text_files/'
IPO_DIR = os.path.join(os.path.expanduser("~"), "Data", "IPO")
BASEDIR = os.path.join(IPO_DIR, "NASDAQ")
FINALJSON = json.loads(open('final_json.txt').read())

YEAR = 2014
QUARTER = 3
IDX_URL = 'ftp://ftp.sec.gov/edgar/full-index/2014/QTR2/master.idx'.format(year=YEAR, Q=QUARTER)
# master = pd.read_csv("/Users/peitalin/Data/IPO/NASDAQ/data/master-2005-2014.idx")
# with open('final_json.txt', 'w') as f:
#     f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))

aget = lambda x: arrow.get(x, 'M/D/YYYY')
firmname = lambda cik: FINALJSON[cik]['Company Overview']['Company Name']
def as_cash(string):
    if '$' not in string:
        return None
    string = string.replace('$','').replace(',','')
    return float(string) if string else None


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
                    'number_of_price_updates'
                    ]
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

        df.loc[cik, 'date_1st_pricing'] = dates[0]
        df.loc[cik, 'date_last_pricing'] = priced_date
        df.loc[cik, 'date_424b'] = B424_date
        df.loc[cik, 'date_trading'] = trade_date

    df['Year'] = [int(2000)]*len(df)
    for cik in df.index:
        df.loc[cik, 'Year'] = aget(FINALJSON[cik]['Company Overview']['Status']).year
    df[float_columns] = df[float_columns].astype(float)
    df['cik'] = df.index
    df.to_csv("price_ranges_temp.csv", dtype={'cik':object})

    return df


def order_df(df):
    cols = [
        'days_to_first_price_update',
        'days_to_final_price_revision',
        'days_from_priced_to_listing',
        'size_of_first_price_update',
        'size_of_final_price_revision',
        'percent_first_price_update',
        'percent_final_price_revision',
        'number_of_price_updates',
        'prange_change_first_price_update',
        'offer_in_filing_price_range',
        'Coname', 'Year',
        'date_1st_pricing', 'date_1st_amendment', 'date_last_pricing',
        'date_424b', 'date_trading',
        'Date', 'Offer Price', 'Open', 'Close', 'High', 'Low', 'Volume',
        'open_return', 'close_return',
        'SIC', 'FF49_industry', '3month_indust_rets', '3month_IPO_volume', 'BAA_yield_changes',
        'confidential_IPO',
        'underwriter_rank_avg', 'underwriter_num_leads', 'underwriter_collective_rank',
        'share_overhang', 'log_proceeds', 'market_cap', 'liab/assets', 'P/E', 'P/sales', 'EPS'
    ]

    col_float = ['days_to_first_price_update', 'days_to_final_price_revision', 'days_from_priced_to_listing', 'size_of_first_price_update', 'size_of_final_price_revision', 'percent_first_price_update', 'percent_final_price_revision', 'number_of_price_updates', 'prange_change_first_price_update', 'Offer Price', 'Open', 'Close', 'High', 'Low', 'Volume', 'open_return', 'close_return', '3month_indust_rets', 'BAA_yield_changes', 'underwriter_rank_avg', 'underwriter_collective_rank', 'share_overhang', 'log_proceeds', 'market_cap', 'liab/assets', 'P/E', 'P/sales', 'EPS']
    col_int = ['Year', 'confidential_IPO', 'underwriter_num_leads', '3month_IPO_volume']
    col_obj = ['offer_in_filing_price_range', 'Coname', 'date_1st_pricing', 'date_1st_amendment', 'date_last_pricing', 'date_424b', 'date_trading', 'Date', 'SIC', 'FF49_industry']

    if not isinstance(df['SIC'].iloc[0], str):
        df["SIC"] = ['0'+str(i) if i<1000 else str(i) for i in df["SIC"]]
    df[col_float] = df[col_float].astype(float)
    df[col_int] = df[col_int].astype(int)
    df[col_obj] = df[col_obj].astype(object)
    return df





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
            df.loc[cik, 'underwriter_rank_avg'] = 0
            df.loc[cik, 'underwriter_num_leads'] = 0
            df.loc[cik, 'underwriter_collective_rank'] = 0
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


def ipo_cycle_variables(df, lag=3):
    """Gets FF49 industry returns, and IPO deal volume
        for each firm in <df> for the <lag> months leading up to an IPO.
    """
    # run VAR to determine optimal lag length of num withdrawn and lag returns
    import pandas.io.data as web
    import requests
    import io
    from zipfile import ZipFile
    from arrow import Arrow

    print("Retrieving Fama-French 49 industry portfolios...")
    FF49 = web.DataReader("49_Industry_Portfolios", "famafrench")[4]
    FFkeys = [re.sub(r'[0-9]{1,2}\s[b]', '', s).replace("'", '') for s in FF49.keys()]
    FF49.columns = FFkeys

    FF49.index = [arrow.get(str(d), 'YYYYMM') for d in FF49.index]
    FFindustries = df['FF49_industry']
    filing_dates = [arrow.get(d[:7], 'YYYY/MM') for d in df['Date']]

    def FF_industry_returns(FFindustries, filing_dates, lag):
        "Indust Returns (Matched on Ken French's industry portfolios)"
        df_returns = []
        for industry, date in zip(FFindustries, filing_dates):
            if date not in FF49.index:
                date = FF49.index[-1]
                lag = 1
            industry_returns = FF49[industry]
            date_range = Arrow.range('month', date.replace(months=(-lag+1)), date)
            df_returns += [sum(industry_returns[d]/100 for d in date_range)/lag]
        return df_returns

    def volume_of_IPO_deals(filing_dates, lag):
        "Gets volume of IPO deals in previous <lag> months, based on df sample"
        monthly_volume = Counter(filing_dates)
        deal_volume = []
        print("Counting volume of IPOs in the %s months before each IPO" % lag)
        for d in df['Date']:
            date = arrow.get(d[:7])
            if date not in FF49.index:
                date = FF49.index[-1]
                lag = 1
            date_range = Arrow.range('month', date.replace(months=(-lag+1)), date)
            deal_volume += [sum([monthly_volume[d] for d in date_range])]
        return deal_volume

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
        for d in df['Date']:
            date = arrow.get(d[:7])
            if date not in FF49.index:
                date = FF49.index[-1]
                lag = 1
            date_range = Arrow.range('month', date.replace(months=-1), date)
            BAA_lagged = BAA[BAA.index.isin(date_range)]
            BAA_yield_changes += [BAA_lagged['BAA'][-1] - BAA_lagged['BAA'][0]]
        return BAA_yield_changes


    print("Matching industry returns on 'FF49_industry'...")
    df['{n}month_indust_rets'.format(n=lag)] = FF_industry_returns(FFindustries, filing_dates, lag)
    df['{n}month_IPO_volume'.format(n=lag)] = volume_of_IPO_deals(filing_dates, lag)
    df['BAA_yield_changes'] = BAA_yield_spread(filing_dates)
    return df


def share_overhang(df):
    as_int = lambda s: float(s.replace(',', ''))
    for cik in df.index:
        shares_outstanding = FINALJSON[cik]['Company Overview']['Shares Outstanding']
        shares_offered = FINALJSON[cik]['Company Overview']['Shares Offered']
        if shares_outstanding==' -- ' or shares_offered==' -- ':
            df.loc[cik, 'share_overhang'] = None
        else:
            df.loc[cik, 'share_overhang'] = as_int(shares_outstanding) / as_int(shares_offered)
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
        print("Getting financial ratios for {}: {}".format(cik, firmname(cik)), end='\r')
        FJ = FINALJSON[cik]

        net_income = as_cash(FJ['Financials']['Net Income'])
        revenue = as_cash(FJ['Financials']['Revenue'])
        liabil = as_cash(FJ['Financials']['Total Liabilities'])
        assets = as_cash(FJ['Financials']['Total Assets'])
        proceeds = as_cash(FJ['Company Overview']['Offer Amount'])
        share_price = as_cash(FJ['Company Overview']['Share Price'])
        num_shares = FJ['Company Overview']['Shares Outstanding'].replace(',', '')

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




## IoI
def cumulative_iot(df, category):
    """Calculates median interest over time (IoT) 60 days before roadshow date,
    CASI (cumulative abnormal search interest) and CSI (cumulative search interest)

    Args:
        --category: 'business-news', 'finance', 'investing', 'all'
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


        if firm['days_to_first_price_update'] == firm['days_to_final_price_revision']:
            end_date = roadshow_start.replace(days=firm['days_to_first_price_update'])
        elif firm['days_to_final_price_revision'] == firm['days_to_final_price_revision']:
            end_date = roadshow_start.replace(days=firm['days_to_final_price_revision'])
        else:
            end_date = arrow.get(firm.date_trading)

        iot_data = pd.read_csv(gtrends_file(cik=cik, category=category))
        iot_triple = list(iot_data[iot_data.columns[:2]].itertuples())
        # get first 2 columns
        iot = [x[2] for x in iot_triple]
        try:
            S = start_index = get_date_index(start_date, iot_triple)
            E = end_index = get_date_index(end_date, iot_triple)
            M = s1_index = get_date_index(s1_date, iot_triple)
            # median IoT in previous 60 days before roadshow commences
            median_iot = median(iot[M-60:M])
            # Cumulative search interest after s1-filing:
            CSI = sum(iot[S:E])
            # Cumulative abnormal search interest:
            CASI = sum([i-median_iot for i in iot if i > median_iot])
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


def weighted_iot(sample, weighting='value_weighted'):

    all_iot = sum([1 for x in sample["IoT_CASI_all"] if x != 0])
    busnews = sum([1 for x in sample["IoT_CASI_business-news"] if x != 0])
    finance = sum([1 for x in sample["IoT_CASI_finance"] if x != 0])
    invest  = sum([1 for x in sample["IoT_CASI_investing"] if x != 0])
    total   = all_iot + busnews + finance + invest

    if weighting=='value_weighted':
        wa = weight_all_iot = all_iot/total
        wb = weight_busnews = busnews/total
        wf = weight_finance = finance/total
        wi = weight_invest  = invest/total
    else:
        wa, wb, wf, wi = [1/4, 1/4, 1/4, 1/4]

    sample['IoT_median_all_finance'] = wa*sample['IoT_median_all'] + wb*sample['IoT_median_business-news'] + wf*sample['IoT_median_finance'] + wi*sample['IoT_median_investing']

    sample['IoT_CASI_all_finance'] = wa*sample['IoT_CASI_all'] + wb*sample['IoT_CASI_business-news'] + wf*sample['IoT_CASI_finance'] + wi*sample['IoT_CASI_investing']

    sample['IoT_CSI_all_finance'] = wa*sample['IoT_CSI_all'] + wb*sample['IoT_CSI_business-news'] + wf*sample['IoT_CSI_finance'] + wi*sample['IoT_CSI_investing']

    sample['ln_CASI_all_finance'] = np.log(sample['IoT_CASI_all_finance'] + 1)

    return sample


def plot_iot(cik, category=''):
    "plots Google Trends Interest-over-time (IoT)"
    from ggplot import ggplot, geom_line, ggtitle, aes

    def gtrends_file(cik, category):
        gtrends_dir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'gtrends-beta', 'cik-ipo', category)
        return os.path.join(gtrends_dir, str(cik)+'.csv')

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

    iot_data['Date'] = [arrow.get(d).date() for d in iot_data['Date']]
    iot_melt = pd.melt(iot_data.icol([0,1]), id_vars=['Date'])
    firm = iot_data.columns[1]
    iot_melt.columns = ['Date', firm, 'Interest-over-time']

    pplot = ggplot(aes(x='Date', y='Interest-over-time', colour=firm), data=iot_melt) + \
            geom_line(color='steelblue') + \
            ggtitle("Interest-over-time for {} - ({})".format(firm, category))

    print(pplot)
    # ggsave(filename='merged_{}'.format(keywords[0].keyword), width=15, height=4)


def process_IOT_variables():
    entities1 = cumulative_iot(df, 'all')
    entities2 = cumulative_iot(df, 'finance')
    entities3 = cumulative_iot(df, 'business-news')
    entities4 = cumulative_iot(df, 'investing')

    df = weighted_iot(df, weighting='equal')

    df['ln_CASI_all'] = np.log(df['IoT_CASI_all'] + 1)
    df['ln_CASI_all_finance'] = np.log(df['IoT_CASI_all_finance'] + 1)
    df['ln_CSI_all'] = np.log(df['IoT_CSI_all'] + 1)
    df['ln_CSI_all_finance'] = np.log(df['IoT_CSI_all_finance'] + 1)

    df.to_csv("attention.csv", index=False)



if __name__=='__main__':

    # Units, ADRs, etc
    # metadata[metadata['Issue Type Code']!='0']
    # FULLJSON = json.loads(open('full_json.txt').read())
    # bad_sic = [
    #     '6021', '6022', '6035', '6036', '6111', '6199', '6153',
    #     '6159', '6162', '6163', '6172', '6189', '6200', '6022',
    #     '6221', '6770', '6792', '6794', '6795', '6798', '6799',
    #     '8880', '8888', '9721', '9995'
    #     ]

    # with open('final_json.txt', 'w') as f:
    #     f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))


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

    cik = '1326801' # Facebook
    cikfb = '1326801' # Facebook


    ## REDO conditional underpricing graphs
    # sb.jointplot(sample.percent_final_price_revision, sample.close_return)





def descriptive_stats():



    keystats = [np.size, np.mean, np.std, np.min, np.median, np.max]
    kkeys = ['offer_in_filing_price_range', 'size_of_first_price_update', 'number_of_price_updates', 'size_of_final_price_revision', 'Year', 'log_proceeds', 'share_overhang', 'EPS',]


    sample2 = sample[kkeys]
    # Industry and Year descriptive stats
    sample2.groupby(['offer_in_filing_price_range', 'Year']).agg(keystats).to_csv("desc_stats.csv")
    sample2.groupby(['offer_in_filing_price_range']).agg(keystats).to_csv("num_price_updates.csv")
    sample2.groupby(['Year']).agg(keystats).to_csv("year_stats.csv")





    # IoI summary stats
    ioi_keys = [x for x in sample.keys() if x.startswith('ioi')] + ["IPO_success"]
    sample[ioi_keys].groupby("IPO_success").describe()



    def price_update_plot():

        plot = plt.plot
        cp = sb.color_palette
        colors = sb.color_palette("muted")
        colors2 = sb.color_palette("husl")

        # from scipy.stats import kendalltau, pearsonr

        laggers = list(sample[sample.days_to_first_price_update > 300].index)
        # sample = sample[sample['days_to_first_price_update'] < 300]


        xy = plt.hist(sample['size_of_first_price_update'],
                    bins=22, alpha=0.6, color=colors2[4], label="N=%s" % len(sample))
        plt.hist(sample['size_of_first_price_update'],
                    bins=44, alpha=0.4, color=colors2[5], label="N=%s" % len(sample))
        plt.xticks(xy[1])
        plt.legend()
        plt.xlim(-12,12)
        plt.ylabel("Frequency")
        plt.xlabel("Size of First Price Update ($)")


        sample2 = df[~df['size_of_final_price_revision'].isnull()]
        # ONLY FOR PLOTTING, 11.5 -> 12
        sample2.loc['1117106', 'size_of_final_price_revision'] = 12
        xy = plt.hist(sample2['size_of_final_price_revision'],
                    bins=24, alpha=0.6, color=colors2[4], label="N=%s" % len(sample2) )
        plt.hist(sample2['size_of_final_price_revision'],
                    bins=48, alpha=0.4, color=colors2[5], label="N=%s" % len(sample2))
        sample2.loc['1117106', 'size_of_final_price_revision'] = 11.5
        plt.legend()
        plt.xticks(xy[1])
        plt.xlim(-12,12)
        plt.ylabel("Frequency")
        plt.xlabel("Size of Price Revision ($)")

        # Upwards price amendmets, and eventual price revision.
        sample3 = sample2[sample2['size_of_first_price_update'] > 0]
        sb.jointplot(
                sample3['percent_first_price_update'],
                sample3['percent_final_price_revision'],
                  # kind='hex'
                  )
        sb.jointplot(
                sample3['size_of_first_price_update'],
                sample3['size_of_final_price_revision'],
                  # kind='hex'
                  )

        # Downwards price amendmets, and eventual price revision.
        sample4 = sample2[sample2.size_of_first_price_update < 0]
        sb.jointplot(
                sample4['percent_first_price_update'],
                sample4['percent_final_price_revision'],
                  # kind='hex'
                  )
        sb.jointplot(
                sample4['size_of_first_price_update'],
                sample4['size_of_final_price_revision'],
                  # kind='hex'
                  )






# def plot_gmail_ipos():
#     co = company[company.Exchange != 'American Stock Exchange']
#     co['proceeds'] = [as_cash(i)/1000000 for i in co['Offer Amount'] if as_cash(i)]
#     plt.hist(co['proceeds'], bins=400)
#     plt.xlim(1,16100)
#     plt.ylim(0,400)
#     plt.ylabel("Frequency")
#     plt.xlabel("Raised proceeds ($Mil)")

#     plt.annotate("Facebook: $16 bil",
#                 (16000, 10),
#                 xytext=(11500, 50),
#                 arrowprops=dict(facecolor=colors[5], width=2, headwidth=2))

#     plt.annotate("Tesla Motors: $226.1 mil",
#                 (226.1, 80),
#                 xytext=(600, 100),
#                 arrowprops=dict(facecolor=colors[3], width=2, headwidth=2))



#     plt.hist([as_cash(s) for s in co['Total Expenses'] if as_cash(s)], bins= 100)
#     plt.xlabel('Fees to underwriter ($Mil)')
#     plt.ylabel('Frequency')

#     plt.hist(df.share_overhang, bins=500)
#     plt.title("Percent Ownership Retained across all IPOs, 2005-2014 Sep")
#     plt.ylabel("Frequency")
#     plt.xlabel("Shares Retained/Shares Offered")
#     plt.annotate("Alibaba IPO: shares_retained/shares_offered = 10x",
#                 (10, 10),
#                 xytext=(10+5, 10+15),
#                 size=11,
#                 arrowprops=dict(facecolor=colors[5], width=2, headwidth=2))
#     plt.xlim(1,50)

# def amihud_plots():

#     filing_count = Counter(metadata['Number of Filings'])
#     filing_count = {1: 1457, 2: 1456, 3: 1452, 4: 1404, 5: 1301, 6: 1127, 7: 847, 8: 579, 9: 358, 10: 220, 11: 139, 12: 73, 13: 48, 14: 36, 15: 19, 16: 13, 17: 9, 18: 8, 19: 6, 20: 4, 21: 4, 22: 4, 23: 3, 24: 3, 25: 3}

#     freq = list(filing_count.values())
#     bins = list(filing_count.keys())
#     data = sum([[b]*n for b,n in zip(bins, freq)], [])
#     plt.hist(data,25)
#     plt.xticks(list(range(1,26)))
#     plt.xlim(1,25)
#     plt.xlabel("No. Filings")
#     plt.ylabel("Frequency")





def delete_old_gtrends(L, remove=False):
    "Finds gtrends files with less than L lines and delets them"

    for ffile in glob(gtdir+'*/*.csv'):
        lines = sum(1 for line in open(ffile))
        if lines < L:
            cik = ffile[-11:-4]
            if remove:
                os.remove(ffile)
            try:
                ipo_date = company.loc[cik, 'Status']
                print("{}: {} <{}> lines => {}".format(cik, firmname(cik), ipo_date, lines))
            except KeyError:
                pass
    print("\n Update these files from Gtrends. ")





def merge_Compustat_FINALJSON(FINALJSON):

    Compustat = pd.read_csv(BASEDIR + "/data/Compustat.csv", dtype=object)
    Compustat.set_index("CIK", inplace=True)

    # GM had multiple securities, identify equity by ticker
    # crspcik = set(Compustat.index)

    # CUSIP: first 6 digits are company identifier
    # next 2 are issue specific, usually numbers for equity, letters for fixed-income
    # 1st issue if usually 10, then 20, 30 ....

    badciks = []
    not_first_offer = []
    ciks = list(set(Compustat.index))
    for cik in ciks:
        if cik not in FINALJSON.keys():
            continue

        TICKER = FINALJSON[cik]['Company Overview']['Proposed Symbol']
        STOCK = Compustat.loc[cik]
        CUSIP = [x for x in set(STOCK['CUSIP']) if x[-3:-1]=='10']
        CONAME = FINALJSON[cik]['Company Overview']['Company Name']
        PRICING_DATE = aget(FINALJSON[cik]['Company Overview']['Status'])

        if len(CUSIP) == 1:
            SP = STOCK[[x in CUSIP for x in STOCK['CUSIP']]]

        elif TICKER in set(STOCK['Ticker']):
            # print('Warning: {} {} may not be 1st equity offering'.format(CONAME, cik))
            SP = STOCK[STOCK['Ticker']==TICKER]
            not_first_offer.append(cik)

        elif CONAME in set(STOCK['Coname']):
            # print('Warning: {} {} may not be 1st equity offering'.format(CONAME, cik))
            SP = STOCK[STOCK['Coname']==CONAME]
            not_first_offer.append(cik)

        elif PRICING_DATE.strftime('%Y/%m/%d') in list(STOCK['Date']):
            CUSIP = list(STOCK[STOCK['Date'] == PRICING_DATE.strftime('%Y/%m/%d')]['CUSIP'])[0]
            SP = STOCK[STOCK['CUSIP'] == CUSIP]

        else:
            print(">> {}: {} did not match on Compustat dataset".format(cik, CONAME))
            badciks.append(cik)


        # WARNING, 1st prices may not be IPO float price
        if isinstance(list(SP['Date'])[0], str):
            SP['Date'] = [arrow.get(d).date() for d in SP['Date']]
        SP = SP[(SP["Date"] - arrow.get(PRICING_DATE).date()) >= 0]
        if len(SP) < 1:
            print("{}: {} no Compustat dates before NASDAQ 'priced date'".format(cik, CONAME))
            badciks.append(cik)
            continue
        FD = next(SP.iterrows())[1]

        FINALJSON[cik]['Metadata']['CUSIP'] = FD["CUSIP"]
        FINALJSON[cik]['Metadata']['GVKEY'] = FD["GVKEY"]
        FINALJSON[cik]['Metadata']['IID']   = FD["IID"]
        FINALJSON[cik]['Metadata']['NAICS'] = FD["NAICS"]
        FINALJSON[cik]['Metadata']['Ticker'] = FD["Ticker"]
        FINALJSON[cik]['Metadata']['SIC'] = FD["SIC"]
        FINALJSON[cik]['Opening Prices'] = {}
        first_day_prices = {'Volume': FD['Volume'],
                            'Close': FD['Close'],
                            'High': FD['High'],
                            'Low': FD['Low'],
                            'Open': FD['Open'],
                            'Date': FD['Date'].strftime('%Y/%m/%d')}
        FINALJSON[cik]['Opening Prices'] = first_day_prices


        # def CRSP_compustat_steps():
            # WRDS -> compustat
            # ciks -> CUSIPS
            # CUSIPS -> CRSP

        # badciks = {cik:firmname(cik) for cik in FINALJSON if 'CUSIP' not in FINALJSON[cik]['Metadata']}
        # cusips = [FINALJSON[cik]['Metadata']['CUSIP'] for cik in FINALJSON]
        # with open("WRDS-cusips.txt", 'w') as f:
        #     write('\n'.join(cusips))
        ### Use these cusips to match with CRSP dataset




def opening_prices_threaded(ciks, N=20):
    "Multi-threaded wrapper for Yahoo Opening Prices"

    def opening_prices(cik):
        "Gets opening prices for IPO"

        ticker = FINALJSON[cik]['Company Overview']['Proposed Symbol']
        coname = FINALJSON[cik]['Company Overview']['Company Name']
        status = FINALJSON[cik]['Company Overview']['Status']
        listing_date = arrow.get(status, 'M/D/YYYY')
        s = listing_date.replace(weeks=-1)
        e = listing_date.replace(weeks=+1)
        print(cik, ticker, coname, listing_date.date())

        url = "https://au.finance.yahoo.com/q/hp"
        query = "?s={ticker}&a={M1}&b={D1}&c={Y1}&d={M2}&e={D2}&f={Y2}&g={freq}"
        url += query.format(ticker=ticker, freq='d',
                            M1=s.month-1, D1=s.day, Y1=s.year,
                            M2=e.month-1, D2=e.day, Y2=e.year)
        # Yahoo month index starts from 0
        headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        sess = requests.get(url)
        html = etree.HTML(sess.text)
        rows = html.xpath('//table[@class="yfnc_datamodoutline1"]/tr/td/table/*')
        if rows:
            prices = rows[-2].xpath('.//text()')
        else:
            prices = ['', '', '', '', '', '', '']

        if ' ' in prices[0]:
            prices[0] = arrow.get(prices[0], 'D/MMM/YYYY').strftime('%Y/%m/%d')

        price_dict = dict(zip(headers, prices))
        price_dict['Volume'] = price_dict['Volume'].replace(',','')
        price_dict.pop("Adj Close")
        return price_dict

    def opening_prices_list(ciks):
        prices_dict = {}
        for cik in ciks:
            prices_dict[cik] = opening_prices(cik)
        return prices_dict

    with ThreadPoolExecutor(max_workers=N) as exec:
        json_result = exec.map(opening_prices_list, [iter(ciks)]*N)
        final_dict  = reduce(lambda d1, d2: dict(d1, **d2), json_result)
    return final_dict
    # pricedict = opening_prices_threaded(ciks, N=20)
    # for cik in pricedict:
    #     FINALJSON[cik]['Opening Prices'] = pricedict[cik]

