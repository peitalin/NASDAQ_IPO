

import csv, glob, json, os
import traceback, time, random
import pandas as pd
import requests
import arrow

from pprint             import pprint
from functools          import reduce
from itertools          import cycle
from urllib.request     import urlopen, Request
from lxml               import etree
from concurrent.futures import ThreadPoolExecutor
from IPython            import embed

N = 10
FILE_PATH = 'text_files/'
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ")
FINALJSON = json.loads(open('final_json.txt').read())


master = pd.read_csv("/Users/peitalin/Data/IPO/NASDAQ/data/master-2005-2014.idx")


YEAR = 2014
QUARTER = 3
IDX_URL = 'ftp://ftp.sec.gov/edgar/full-index/2014/QTR2/master.idx'.format(year=YEAR, Q=QUARTER)



## Underwriters
def underwriter_ranks(df):
    """Gets the Ritter/Carter-Manaster rank for the 'lead_underwriters' in the supplied dataframe
    dataframe musthave keys: lead_underwriter and s1_date
    """

    from functools import partial
    from arrow     import Arrow
    uw_rank = pd.read_csv("SEC_index/uw_rank.csv", encoding="latin-1")
    uw_rank = uw_rank.dropna()
    uw_rank = uw_rank.set_index("Underwriter")
    adate = arrow.get

    def is_same_alphabet(a_reference, a):
        return a.lower().startswith(a_reference)


    # Fuzzy match to Carter-Manaster underwriter names
    lookup_uw = {'No Underwriter': 'No Underwriter',
                 'Self-underwritten': 'Self-underwritten',
                 ' -- ': 'No Underwriter'}

    for uw in set(sample['lead_underwriter']):
        if uw in ['No Underwriter', 'Self-underwritten', ' -- ']:
            continue

        is_same_alpha = partial(is_same_alphabet, uw[0].lower())
        uw_match_set  = set(filter(is_same_alpha, uw_rank.index))
        matched_uw = process.extractOne(uw, uw_match_set)
        print(uw, '->', matched_uw)
        lookup_uw.update({uw:matched_uw[0]})


    underwriter_ranks = []
    for cik, uw, s1_date in df[['lead_underwriter', 's1_date']].itertuples():
        if uw in ['No Underwriter', 'Self-underwritten', ' -- ']:
            underwriter_ranks += [-1]
            continue

        if adate(s1_date) < adate('2007-01-01'):
            rank_year = '2005-2007'
        elif adate(s1_date) > adate('2010-01-01'):
            rank_year = '2010-2011'
        else:
            rank_year = '2008-2009'

        uw_c_manaster = lookup_uw[uw]
        cm_rank = uw_rank.loc[uw_c_manaster, rank_year]
        print(uw, '->', uw_c_manaster, '\n\trank:', cm_rank)
        underwriter_ranks += [round(cm_rank, 1)]

    df['lead_underwriter_rank'] = underwriter_ranks
    return df



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



def opening_prices_threaded(ciks, N=20):
    "Multi-threaded wrapper for Yahoo Opening Prices"

    def opening_prices_list(ciks):
        prices_dict = {}
        for cik in ciks:
            prices_dict[cik] = opening_prices(cik)
        return prices_dict

    with ThreadPoolExecutor(max_workers=N) as exec:
        json_result = exec.map(opening_prices_list, [iter(ciks)]*N)
        final_dict  = reduce(lambda d1, d2: dict(d1, **d2), json_result)
    return final_dict



def fix_ciks_7_digits():
    ciks = list(FINALJSON.keys())
    for cik in ciks:
        if len(cik)==5:
            newcik = '00' + cik
        elif len(cik)==6:
            newcik = '0' + cik
        else:
            continue
        FINALJSON[newcik] = FINALJSON[cik].copy()
        FINALJSON[newcik]['Company Overview']['CIK'] = newcik
        FINALJSON.pop(cik)


def merge_CRSP_FINALJSON():
    CRSP = pd.read_csv("CRSP.csv", dtype=object)
    # pricedict = opening_prices_threaded(ciks, N=20)
    # for cik in pricedict:
    #     FINALJSON[cik]['Opening Prices'] = pricedict[cik]

    CRSP2 = pd.read_csv("data/CRSP2_openingprices.csv", dtype=object)
    CRSP2.set_index("CIK", inplace=True)

    # GM had multiple securities, identify equity by ticker

    # crspcik = set(CRSP.index)

    # CUSIP: first 6 digits are company identifier
    # next 2 are issue specific, usually numbers for equity, letters for fixed-income
    # 1st issue if usually 10, then 20, 30 ....

    badciks = []
    not_first_offer = []
    for cik in set(CRSP.index):
        if cik not in FINALJSON.keys():
            continue

        TICKER = FINALJSON[cik]['Company Overview']['Proposed Symbol']
        STOCK = CRSP.loc[cik]
        CUSIP = [x for x in set(STOCK['CUSIP']) if x[-3:-1]=='10']
        CONAME = FINALJSON[cik]['Company Overview']['Company Name']

        if len(CUSIP) == 1:
            SP = STOCK[[x in CUSIP for x in STOCK['CUSIP']]]
        elif TICKER in set(STOCK['Ticker']):
            print('Warning: {} {} may not be 1st equity offering'.format(CONAME, cik))
            SP = STOCK[STOCK['Ticker']==TICKER]
            not_first_offer.append(cik)
        elif CONAME in set(STOCK['Coname']):
            print('Warning: {} {} may not be 1st equity offering'.format(CONAME, cik))
            SP = STOCK[STOCK['Coname']==CONAME]
            not_first_offer.append(cik)
        else:
            IPO_DATE = arrow.get(FINALJSON[cik]['Company Overview']['Status'], 'M/D/YYYY')
            IPO_DATE = IPO_DATE.strftime('%Y/%m/%d')
            if IPO_DATE in list(STOCK['Date']):
                CUSIP = list(STOCK[STOCK['Date'] == IPO_DATE]['CUSIP'])[0]
            else:
                print(">> {}: {} did not match on CRSP dataset".format(cik, CONAME))
                badciks.append(cik)

        FD = next(SP.iterrows())[1]
        FINALJSON[cik]['Metadata']['CUSIP'] = FD["CUSIP"]
        FINALJSON[cik]['Metadata']['GVKEY'] = FD["GVKEY"]
        FINALJSON[cik]['Metadata']['IID']   = FD["IID"]
        FINALJSON[cik]['Metadata']['NAICS'] = FD["NAICS"]
        FINALJSON[cik]['Metadata']['Ticker'] = FD["Ticker"]
        FINALJSON[cik]['Opening Prices'] = {}
        first_day_prices = {'Volume': FD['Volume'],
                            'Close': FD['Close'],
                            'High': FD['High'],
                            'Low': FD['Low'],
                            'Open': FD['Open'],
                            'Date': FD['Date']}
        FINALJSON[cik]['Opening Prices'] = first_day_prices


    def check_CRSP_aligns_IPO_date():

        missing_ciks = set(FINALJSON.keys()) - set(CRSP.index)

        still_bad_ciks = []
        for cik in missing_ciks:
            coname = FINALJSON[cik]['Company Overview']['Company Name']
            ipo_date = FINALJSON[cik]['Filing'][0][2]
            ipo_date = arrow.get(ipo_date, 'M/D/YYYY')
            start, end = ipo_date.replace(months=-1), ipo_date.replace(months=+1)

            try:
                crspdate = FINALJSON[cik]['Opening Prices']['Date']
                if crspdate == '':
                    badciks.append(cik)
                    print("No date for {}: {}".format(cik, coname))
                    continue
                try:
                    crspdate2 = arrow.get(crspdate, 'YYYY/M/D')
                except:
                    crspdate2 = arrow.get(crspdate, 'D/MMM/YYYY')

                if not start < crspdate2 < end:
                    new_open_prices = opening_prices(cik)
                    if new_open_prices['Date'] != '':
                        if arrow.get(new_open_prices['Date'], 'YYYY/M/D') == ipo_date:
                            FINALJSON[cik]['Opening Prices'] = new_open_prices
                            continue
                    print("No price data for {}: {}".format(cik, coname))
                    still_bad_ciks.append(cik)

            except KeyError:
                new_open_prices = opening_prices(cik)
                if new_open_prices['Date'] != '':
                    if arrow.get(new_open_prices['Date'], 'YYYY/M/D') == ipo_date:
                        FINALJSON[cik]['Opening Prices'] = new_open_prices
                        continue
                print("No price data for {}: {}".format(cik, coname))
                still_bad_ciks.append(cik)

        # [FINALJSON[cik]['Company Overview']['Company Name'] for cik in still_bad_ciks]
        # [FINALJSON.pop(cik) for cik in still_bad_ciks]
        # shitciks = [FINALJSON[cik]['Company Overview']['Company Name'] for cik in FINALJSON.keys()
        #             if FINALJSON[cik]['Opening Prices']['Volume'] == "" ]
        # [FINALJSON.pop(cik) for cik in shitciks]
