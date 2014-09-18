

import csv, glob, json, os, re
import traceback, time, random
import pandas as pd
import numpy as np
import requests
import arrow
import matplotlib.pyplot as plt
import seaborn as sb

from pprint             import pprint
from functools          import reduce
from itertools          import cycle
from lxml               import etree
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

aget = lambda x: arrow.get(x, 'M/D/YYYY')
firmname = lambda cik: FINALJSON[cik]['Company Overview']['Company Name']
def as_cash(string):
    if '$' not in string:
        return None
    string = string.replace('$','').replace(',','')
    return float(string) if string else None



## Underwriters
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
        if uw_list != uw_list:
            df.loc[cik, 'lead_underwriter_rank'] = None
            continue
        if any(uw in na_uw for uw in uw_list):
            df.loc[cik, 'lead_underwriter_rank'] = None
            continue

        uw_ranks = [get_uw_rank(uw,cik) for uw in uw_list]
        # #Pick top underwriter rank
        # CM_rank = round(max(uw_ranks), 1)
        # #Average underwriter rank
        CM_rank = round(sum(uw_ranks)/len(uw_list), 1)
        df.loc[cik, 'lead_underwriter_rank'] = CM_rank
    # df.to_csv("price_ranges.csv", dtype={'cik':object})



## IoI
def cumulative_ioi(category, df):
    """Calculates median interest over time 60 days before s1-filing date,
    CASI (cumulative abnormal search interest) and CSI (cumulative search interest)

    Args:
        --category: 'business-news', 'finance', 'investing', 'all'
        --df= dataframe
    """

    from numpy import median

    def gtrends_files(category, df):
        gtrends_dir = os.path.join(os.path.expanduser('~'), \
                        'Dropbox', 'gtrends-beta', 'cik-ipo', category)
        return [os.path.join(gtrends_dir, str(cik)+'.csv') for cik in df.cik]

    def get_date_index(ref_date, ioi_triple):
        for triple in ioi_triple:
            index, date, ioi = triple
            if arrow.get(date) < arrow.get(ref_date):
                continue
            else:
                return index
        return len(ioi_triple) - 1


    median_ioi_60days, CASI, CSI = [], [], []
    zip_args = zip(gtrends_files(category, df),
                   df['s1_date'],
                   df['exit_date'])

    for ioi_file, s1_date, exit_date in list(zip_args):
        print('\rCrunching interest-over-time metrics', ioi_file, ' '*40, end='\r')
        ioi_data = pd.read_csv(ioi_file)
        ioi_triple = list(ioi_data[ioi_data.columns[:2]].itertuples())
        # get first 2 columns
        ioi = [x[2] for x in ioi_triple]
        try:
            s1_index  = get_date_index(s1_date, ioi_triple)
            end_index = get_date_index(exit_date, ioi_triple)
            # median ioi in previous 60 days before s1 filing
            median_ioi = median(ioi[s1_index-60:s1_index])
            # Cumulative search interest after s1-filing:
            CSI_  = sum(ioi[s1_index:end_index])


            # If IPO takes longer than 15 weeks, count only last 15 weeks
            # Otherwise delayed IPOs get large attention scores.
            if end_index - s1_index > 4*4*7:
                # 4 months * 4 weeks * 7 days = 112 days
                ioi = ioi[-112:]
            else:
                ioi = ioi[s1_index:end_index]

            # Cumulative abnormal search interest:
            CASI_ = sum([i-median_ioi for i in ioi if i-median_ioi > 0])
            # embed()
        except: # (ValueError, IndexError) as e:
            print(s1_date, 'not in', ioi_triple[0][0], "-", ioi_triple[-1][0])
            median_ioi = 0
            CSI_  = 0
            CASI_ = 0

        if median_ioi != median_ioi:
            median_ioi, CSI_, CASI_ = 0, 0, 0

        median_ioi_60days += [median_ioi]
        CASI += [CASI_]
        CSI  += [CSI_]

    return median_ioi_60days, CASI, CSI

def weighted_ioi(sample, weighting='value_weighted'):

    all_ioi = sum([1 for x in sample.ioi_CASI_all if x != 0])
    busnews = sum([1 for x in sample.ioi_CASI_busnews if x != 0])
    finance = sum([1 for x in sample.ioi_CASI_finance if x != 0])
    invest  = sum([1 for x in sample.ioi_CASI_investing if x != 0])
    total   = all_ioi + busnews + finance + invest

    if weighting=='value_weighted':
        wa = weight_all_ioi = all_ioi/total
        wb = weight_busnews = busnews/total
        wf = weight_finance = finance/total
        wi = weight_invest  = invest/total
    else:
        wa, wb, wf, wi = [1/4, 1/4, 1/4, 1/4]

    sample['ioi_median_all_finance'] = wa*sample['ioi_median_all'] + wb*sample['ioi_median_busnews'] + wf*sample['ioi_median_finance'] + wi*sample['ioi_median_investing']

    sample['ioi_CASI_all_finance'] = wa*sample['ioi_CASI_all'] + wb*sample['ioi_CASI_busnews'] + wf*sample['ioi_CASI_finance'] + wi*sample['ioi_CASI_investing']

    sample['ioi_CSI_all_finance'] = wa*sample['ioi_CSI_all'] + wb*sample['ioi_CSI_busnews'] + wf*sample['ioi_CSI_finance'] + wi*sample['ioi_CSI_investing']

    sample['ln_CASI_all_finance'] = np.log(sample['ioi_CASI_all_finance'] + 1)

    return sample





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

    colnames = ['days_to_first_price_update',
                'days_to_final_price_revision',
                'days_from_priced_to_listing',
                'size_of_first_price_update',
                'size_of_final_price_revision',
                'percent_first_price_update',
                'percent_final_price_revision',
                'offer_in_filing_price_range',
                'prange_change_first_price_update',
                'number_of_price_updates',
                ]
    df = pd.DataFrame([[0]*len(colnames)]*len(filings), filings.index,
                    columns=colnames, dtype=np.float64)


    RW, MISSING, ABNORMAL = [], [], set()
    # cik, values = '1326801', filings.loc['1326801']
    # cik, values = '1472595', filings.loc['1472595']
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
                ABNORMAL |= {cik}

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
            in_range = filing_price_range[0]-1 < listing_price < filing_price_range[0]+1
            offer_in_filing_price_range = in_range
        else:
            in_range = min(filing_price_range) < listing_price < max(filing_price_range)
            offer_in_filing_price_range = in_range

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
                        ABNORMAL |= {cik}
                        print("{}: {} has large price updates".format(cik, firmname(cik)))

            number_of_price_updates = num_price_updates(prices)
            days_to_final_price_revision = (final_rev_date - dates[0]).days
            size_of_final_price_revision = listing_price - prices[0]
            percent_final_price_revision = (listing_price - prices[0]) / prices[0]


        df.loc[cik, 'days_to_first_price_update'] = days_to_first_price_update
        df.loc[cik, 'size_of_first_price_update'] = size_of_first_price_update
        df.loc[cik, 'percent_first_price_update'] = percent_first_price_update

        df.loc[cik, 'days_to_final_price_revision'] = days_to_final_price_revision
        df.loc[cik, 'size_of_final_price_revision'] = size_of_final_price_revision
        df.loc[cik, 'percent_final_price_revision'] = percent_final_price_revision

        df.loc[cik, 'days_from_priced_to_listing'] = (final_rev_date - trade_date).days
        df.loc[cik, 'offer_in_filing_price_range'] = offer_in_filing_price_range
        df.loc[cik, 'prange_change_first_price_update'] = prange_change_first_price_update
        df.loc[cik, 'number_of_price_updates'] = number_of_price_updates

    df['Year'] = [int(2000)]*len(df)
    for cik in df.index:
        df.loc[cik, 'Year'] = aget(FINALJSON[cik]['Company Overview']['Status']).year
    df[colnames] = df[colnames].astype(float)
    df['cik'] = df.index
    df.to_csv("price_ranges.csv", dtype={'cik':object})


def match_sic_indust_FF(df):
    """Matches df['SIC'] with the industry categories listed in
    one of Kenneth French's industry portfolios.
    """

    import pandas.io.data as web
    import requests
    import io
    from zipfile import ZipFile

    FF49 = web.DataReader("49_Industry_Portfolios", "famafrench")[4]
    FFkeys = [re.sub(r'[0-9]{1,2}\s[b]', '', s).replace("'", '') for s in FF49.keys()]
    FF49.columns = FFkeys

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


def ipo_cycle_variables(df, lag=3):
    # run VAR to determine optimal lag length of num withdrawn and lag returns

    s1_file_key = 's1_date' if "s1_date" in df.keys() else 'date'
    indust = pd.read_csv("SEC_index/12_Industry_Portfolios.csv")
    indust.set_index("Date", inplace=True)
    indust.index = [arrow.get(d) for d in indust.index]
    ff_indust = df['industry_FF12']
    filing_date = [arrow.get(d[:7], 'YYYY-MM') for d in df[s1_file_key]]

    # Indust Returns (Matched on Ken French's industry portfolios)
    df_ret = []
    for ff_ind, date in zip(ff_indust, filing_date):
        ind_ret = indust[ff_ind]
        date_range = arrow.Arrow.range('month', date.replace(months=(-lag+1)), date)
        df_ret.append(sum([ind_ret[d] for d in date_range])/lag)

    # Num withdrawn
    from collections import Counter
    monthly_withdrawn = Counter(filing_date)
    num_withdraw = []
    for d in df[s1_file_key]:
        date = arrow.get(d[:7])
        date_range = arrow.Arrow.range('month', date.replace(months=(-lag+1)), date)
        num_withdraw += [sum([monthly_withdrawn[d] for d in date_range])]

    # Number of IPOs (IPO deal volume)
    IPO_scoop = pd.read_csv("SEC_index/archives/Gtrends_IPOscoop.csv")
    num_IPO_per_month = Counter([arrow.get(d[:7], 'YYYY-MM') for d in IPO_scoop['Date']])
    num_IPOs = []
    for d in df[s1_file_key]:
        date = arrow.get(d[:7])
        date_range = arrow.Arrow.range('month', date.replace(months=(-lag+1)), date)
        num_IPOs += [sum([num_IPO_per_month[d] for d in date_range])]

    df['{n}month_indust_rets'.format(n=lag)] = df_ret
    df['{n}month_withdrawn'.format(n=lag)] = num_withdraw
    df['{n}month_IPO_volume'.format(n=lag)] = num_IPOs
    return df

def BAA_yield_spread(sample):
    """Calculates changes in the BAA corporate bond yield spread (Moody's)
    between s1_date (-1 week) and exit/withdrawal date
    """

    BAAYLD = pd.read_csv("IPO_cycles/WBAAYLD.csv").sort("Date").set_index("Date")[2200:]
    BAAYLD.index = [arrow.get(d).date() for d in BAAYLD.index]

    BAA_yield_change = []
    BAA_yield_volatility = []

    for s1, e1 in sample[['s1_date', 'exit_date']].itertuples(index=False):
        s1, e1 = arrow.get(s1).replace(weeks=-1), arrow.get(e1)
        date_range = arrow.Arrow.range('day',s1,e1)
        BAA = BAAYLD[BAAYLD.index.isin(date_range)]
        BAA_yield_change += [BAA['Value'][0] - BAA['Value'][-1]]
        BAA_yield_volatility += [BAA['Value'].std()**2]

    sample['BAA_yield_change'] = BAA_yield_change
    sample['BAA_yield_volatility'] = BAA_yield_volatility
    return sample






if __name__=='__main__':

    FINALJSON = json.loads(open('final_json.txt').read())
    ciks = sorted(FINALJSON.keys())

    company     = pd.DataFrame([FINALJSON[cik]['Company Overview']  for cik in ciks], ciks)
    financials  = pd.DataFrame([FINALJSON[cik]['Financials']        for cik in ciks], ciks)
    experts     = pd.DataFrame([FINALJSON[cik]['Experts']           for cik in ciks], ciks)
    metadata    = pd.DataFrame([FINALJSON[cik]['Metadata']          for cik in ciks], ciks)
    filings     = pd.DataFrame([FINALJSON[cik]['Filing']            for cik in ciks], ciks)
    open_prices = pd.DataFrame([FINALJSON[cik]['Opening Prices']    for cik in ciks], ciks)


    df = pd.read_csv('price_ranges.csv', dtype={'cik':object})
    df.set_index('cik', inplace=True)

    cik = '1326801' # Facebook



def descriptive_stats():
    # df_filings()
    df = pd.read_csv('price_ranges.csv', dtype={'cik':object})
    df.set_index('cik', inplace=True)
    # df.sort_index(inplace=True)
    # df = df.join(open_prices.sort_index())
    # cols = ['Open', 'Close', 'High', 'Low', 'Volume']
    # df['Offer Price'] = [as_cash(s) for s in company.sort_index()['Share Price']]
    # df = df[colnames + ['Year', 'Date', 'Offer Price'] + cols]
    # df[cols] = df[cols].astype(float)
    # df['open_return'] = (df['Open'] - df['Offer Price']) / df['Offer Price']
    # df['close_return'] = (df['Close'] - df['Offer Price']) / df['Offer Price']
    # df["SIC"] = metadata["SIC"]
    # df['Coname'] = company['Company Name']

    # df.to_csv("price_ranges.csv", dtype={'cik':object})


    sample = df[~df.size_of_first_price_update.isnull()]

    sb.jointplot(
            sample["size_of_final_price_revision"],
            sample["close_return"],
             # kind='hex'
             )

    ##############################################
    # REDO price updates: PERCENT price updates
    ##############################################


    keystats = [np.size, np.mean, np.std, np.min, np.median, np.max]

    sample = sample[['size_of_first_price_update', 'number_of_price_updates',
                    'size_of_final_price_revision', 'Year']]
    # Industry and Year descriptive stats
    sample.groupby(['number_of_price_updates', 'Year']).agg(keystats).to_csv("desc_stats.csv")
    sample.groupby(['number_of_price_updates']).agg(keystats).to_csv("num_price_updates.csv")
    sample.groupby(['Year']).agg(keystats).to_csv("year_stats.csv")

    # kkeys = ['3month_IPO_volume', '3month_indust_rets', '3month_withdrawn', 'IPO_duration', 'lead_underwriter_rank', 's1_aggregate_offering_proceeds']


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
        sample = sample[sample['days_to_first_price_update'] < 300]

        sample.loc['1161448', 'size_of_first_price_update'] = 12
        # ONLY FOR PLOTTING, 12.5 -> 12
        xy = plt.hist(sample['size_of_first_price_update'],
                    bins=24, alpha=0.6, color=colors2[4], label="N=%s" % len(sample))
        plt.hist(sample['size_of_first_price_update'],
                    bins=48, alpha=0.4, color=colors2[5], label="N=%s" % len(sample))
        plt.xticks(xy[1])
        plt.legend()
        plt.xlim(-12,12)
        plt.ylabel("Frequency")
        plt.xlabel("Size of First Price Update ($)")
        # sample.loc['1161448', 'size_of_first_price_update'] = 12.5


        sample2 = df[~df['size_of_final_price_revision'].isnull()]
        sample2.loc['1161448', 'size_of_final_price_revision'] = 19
        xy = plt.hist(sample2['size_of_final_price_revision'],
                    bins=31, alpha=0.6, color=colors2[4])
        plt.hist(sample2['size_of_final_price_revision'],
                    bins=62, alpha=0.4, color=colors2[5])
        plt.legend()
        plt.xticks(xy[1])
        plt.xlim(-12,12)
        plt.ylabel("Frequency")
        plt.xlabel("Size of Price Revision ($)")
        sample2.loc['1161448', 'size_of_first_price_update'] = 19.5


        sb.jointplot(
                sample.size_of_first_price_update,
                sample.size_of_final_price_revision,
                  kind='hex')



    # 1) time between filing and 1st pricing
    # 2) time between 1st pricing and 1st amendment
    # 3) number of amendments
    # 4) time between 1st pricing and listing date (roadshow duration)
    # 5) absolute price range

    # → group by upwards, within file range and downwards amendments
    # → group by time from issue (or time from 1st pricing and 1st amendment






def amihud_plots():


    # sb.set_palette("deep", desat=.6)
    # sb.set_context(rc={"figure.figsize": (8, 4)})

    # freq = [142,9,9,7,5,3,3,2,4,2,3,3,1,4,2,4,0,2,6,73]
    # bins = [x for x in np.linspace(0,1,21)]
    # xlab = [str(x)+' ~ '+str(y) for x,y in zip(np.linspace(0,1,20), bins)]


    # data = sum([[b]*n for b,n in zip(np.linspace(0,1,20), freq)], [])

    # plt.hist(data,20)
    # plt.xticks(np.linspace(0,1, 21))
    # plt.xlabel("Allocation Rate (%)")
    # plt.ylabel("Frequency")

    import matplotlib.pyplot as plt
    from collections import Counter

    filing_count = Counter(metadata['Number of Filings'])
    filing_count = {1: 1457, 2: 1456, 3: 1452, 4: 1404, 5: 1301, 6: 1127, 7: 847, 8: 579, 9: 358, 10: 220, 11: 139, 12: 73, 13: 48, 14: 36, 15: 19, 16: 13, 17: 9, 18: 8, 19: 6, 20: 4, 21: 4, 22: 4, 23: 3, 24: 3, 25: 3}

    freq = list(filing_count.values())
    bins = list(filing_count.keys())
    data = sum([[b]*n for b,n in zip(bins, freq)], [])
    plt.hist(data,25)
    plt.xticks(list(range(1,26)))
    plt.xlim(1,25)
    plt.xlabel("No. Filings")
    plt.ylabel("Frequency")




    # share overhang
    # SIC industry returns
    # IPO cycle variables
    # Gtrends variables
    # underwriter rank (average lead underwriters)
    # no. underwriters
    # VC dummy (crunchbase)
    # confidential IPO
    # EPS



    # The IPO specific control variables are:
    # (1) Up revision–percentage upward revision from the mid-point of the filing range if the offer price is greater than the mid-point, else zero;
    # (2) VC dummy–a dummy variable set to one if the IPO is backed by venture capital, else zero;
    # (3) Top-tier dummy–a dummy variable set to one if the IPO’s lead underwriter has a value of eight or more using Carter and Manaster (1990)
    # (4) Positive EPS dummy–a dummy variable set to one if the IPO has positive earnings per share (EPS) in the 12 months prior to going public, else zero;
    # (5) Prior Nasdaq 15-day returns–the buy-and-hold returns
    # (6) Share overhang–the number of shares retained divided by the number of shares in the initial offering;
    # (7) Sales–trailing firm annual sales in millions of dollars.

    # shares outstanding / shares offered



    # ioi_output_triple = cumulative_ioi('all', sample)
    # sample['ioi_median_all'] = ioi_output_triple[0]
    # sample['ioi_CASI_all']   = ioi_output_triple[1]
    # sample['ioi_CSI_all']    = ioi_output_triple[2]

    # ioi_output_triple = cumulative_ioi('business-news', sample)
    # sample['ioi_median_busnews'] = ioi_output_triple[0]
    # sample['ioi_CASI_busnews']   = ioi_output_triple[1]
    # sample['ioi_CSI_busnews']    = ioi_output_triple[2]

    # ioi_output_triple = cumulative_ioi('finance', sample)
    # sample['ioi_median_finance'] = ioi_output_triple[0]
    # sample['ioi_CASI_finance']   = ioi_output_triple[1]
    # sample['ioi_CSI_finance']    = ioi_output_triple[2]

    # ioi_output_triple = cumulative_ioi('investing', sample)
    # sample['ioi_median_investing'] = ioi_output_triple[0]
    # sample['ioi_CASI_investing']   = ioi_output_triple[1]
    # sample['ioi_CSI_investing'] = ioi_output_triple[2]


    # sample = weighted_ioi(sample, weighting='equal')

    # sample['ln_CASI_all'] = np.log(sample['ioi_CASI_all'] + 1)
    # sample['ln_CASI_all_finance'] = np.log(sample['ioi_CASI_all_finance'] + 1)
    # sample['ln_CSI_all'] = np.log(sample['ioi_CSI_all'] + 1)
    # sample['ln_CSI_all_finance'] = np.log(sample['ioi_CSI_all_finance'] + 1)

    # sample.to_csv("SEC_index/attention.csv", index=False)



























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




def merge_CRSP_FINALJSON():
    CRSP = pd.read_csv(BASEDIR + "/data/CRSP.csv", dtype=object)
    CRSP.set_index("CIK", inplace=True)

    # GM had multiple securities, identify equity by ticker
    # crspcik = set(CRSP.index)

    # CUSIP: first 6 digits are company identifier
    # next 2 are issue specific, usually numbers for equity, letters for fixed-income
    # 1st issue if usually 10, then 20, 30 ....

    badciks = []
    not_first_offer = []
    ciks = list(set(CRSP.index))
    for cik in ciks:
        if cik not in FINALJSON.keys():
            continue

        TICKER = FINALJSON[cik]['Company Overview']['Proposed Symbol']
        STOCK = CRSP.loc[cik]
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
            print(">> {}: {} did not match on CRSP dataset".format(cik, CONAME))
            badciks.append(cik)


        # WARNING, 1st prices may not be IPO float price
        if isinstance(list(SP['Date'])[0], str):
            SP['Date'] = [arrow.get(d).date() for d in SP['Date']]
        SP = SP[(SP["Date"] - arrow.get(PRICING_DATE).date()) >= 0]
        if len(SP) < 1:
            print("{}: {} no CRSP dates before NASDAQ 'priced date'".format(cik, CONAME))
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

        # [FINALJSON[cik]['Company Overview']['Company Name'] for cik in still_bad_ciks]
        # [FINALJSON.pop(cik) for cik in still_bad_ciks]
        # shitciks = [FINALJSON[cik]['Company Overview']['Company Name'] for cik in FINALJSON.keys()
        #             if FINALJSON[cik]['Opening Prices']['Volume'] == "" ]
        # [FINALJSON.pop(cik) for cik in shitciks]

       # {FINALJSON[cik]['Metadata']['SIC']:FINALJSON[cik]['Metadata']['SIC code'] for cik in FINALJSON}
