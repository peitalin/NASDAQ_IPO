

import glob, json, os, re
import pandas as pd
import numpy as np
import requests
import arrow
import matplotlib.pyplot as plt
import seaborn as sb

from widgets            import as_cash, firmname, get_cik, write_FINALJSON
from collections        import Counter, OrderedDict
from pprint             import pprint
from concurrent.futures import ThreadPoolExecutor


N = 10
FILE_PATH = 'text_files/'
IPO_DIR = os.path.join(os.path.expanduser("~"), "Data", "IPO")
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ",)
FILEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "Filings")
FINALJSON = json.loads(open('final_json.txt').read())
FULLJSON = json.loads(open('full_json.txt').read())
# with open('full_json.txt', 'w') as f:
#     f.write(json.dumps(FULLJSON, indent=4, sort_keys=True))

YEAR = 2014
QUARTER = 3
IDX_URL = 'ftp://ftp.sec.gov/edgar/full-index/{year}/QTR{Q}/master.idx'.format(year=YEAR, Q=QUARTER)
# master = pd.read_csv("/Users/peitalin/Data/IPO/NASDAQ/data/master-2005-2014.idx")
# with open('final_json.txt', 'w') as f:
#     f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))






def merge_Compustat_FINALJSON(FULLJSON=FULLJSON, compustat_file="/data/Compustat.csv"):
    """Merges NASDAQ data with Compustate data by CIK.
    Where this fails, matches by Company name, Ticker and IPO listing dates."""

    Compustat = pd.read_csv(BASEDIR + compustat_file, dtype=object)
    Compustat.set_index("CIK", inplace=True)

    # CUSIP: first 6 digits are company identifier
    # next 2 are issue specific, usually numbers for equity, letters for fixed-income
    # 1st issue if usually 10, then 20, 30 ....

    badciks = []
    not_first_offer = []
    ciks = list(set(Compustat.index))
    for cik in ciks:
        if cik not in FULLJSON.keys():
            continue

        TICKER = FULLJSON[cik]['Company Overview']['Proposed Symbol']
        STOCK = Compustat.loc[cik]
        CUSIP = [x for x in set(STOCK['CUSIP']) if x[-3:-1]=='10']
        CONAME = FULLJSON[cik]['Company Overview']['Company Name']
        PRICING_DATE = aget(FULLJSON[cik]['Company Overview']['Status'])

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

        FULLJSON[cik]['Metadata']['CUSIP'] = FD["CUSIP"]
        FULLJSON[cik]['Metadata']['GVKEY'] = FD["GVKEY"]
        FULLJSON[cik]['Metadata']['IID']   = FD["IID"]
        FULLJSON[cik]['Metadata']['NAICS'] = FD["NAICS"]
        FULLJSON[cik]['Metadata']['Ticker'] = FD["Ticker"]
        FULLJSON[cik]['Metadata']['SIC'] = FD["SIC"]
        FULLJSON[cik]['Opening Prices'] = {}
        first_day_prices = {'Volume': FD['Volume'],
                            'Close': FD['Close'],
                            'High': FD['High'],
                            'Low': FD['Low'],
                            'Open': FD['Open'],
                            'Date': FD['Date'].strftime('%Y/%m/%d')}
        FULLJSON[cik]['Opening Prices'] = first_day_prices

        # badciks = {cik:firmname(cik) for cik in FULLJSON if 'CUSIP' not in FULLJSON[cik]['Metadata']}
        # cusips = [FULLJSON[cik]['Metadata']['CUSIP'] for cik in FULLJSON]
        # with open("WRDS-cusips.txt", 'w') as f:
        #     write('\n'.join(cusips))
        ### Use these cusips to match with CRSP dataset



def get_compustat_metadata(FULLJSON=FULLJSON):
    """Uses CUSIP ids to identify stock issue type (ETF, common shares, preferred, units, ADRs, etc).
    FULLJSON: nasdaq IPO data in JSON format.
    """

    compustat = pd.read_csv("data/compustat-id.csv")
    compustat.set_index("cusip", inplace=True)
    # tpci: Issue Type
    # stko: Stock Ownership Code: public firm or subsidiary or LBO
    issue_types = {
        '%': 'ETF',
        '0': 'Common Ordinary Shares',
        '1': 'Preferred Shares',
        '2': 'Warrant/Right',
        '4': 'Unit',
        'F': 'ADR',
        'G': 'Convertible Preferred',
        'R': 'Structured Product',
        'S': 'General Service Administration',
    }
    ownership_types = {
        0: 'Public Company',
        1: 'Subsidiary of public company',
        2: 'Subsidiary of private company',
        3: 'Public company of small exchange', # OTCBB, pinkslips
        4: 'Levered Buyout'
    }

    compustat_cutoff_date = arrow.get('2014-08-24')

    badciks = []
    for cik in FULLJSON:
        try:
            cusip = FULLJSON[cik]['Metadata']['CUSIP']
            assert len(cusip) == 9
            tpci = compustat.loc[cusip]['tpci'].iloc[0]
            FULLJSON[cik]['Metadata']['Issue Type Code'] = tpci
            FULLJSON[cik]['Metadata']['Issue Type'] = issue_types[tpci]
        except KeyError:
            ipo_date = FULLJSON[cik]['Company Overview']['Status']
            print("{}: {} {}".format(cik, firmname(cik), ipo_date))
            badciks.append(cik)
    return FULLJSON



def opening_prices_threaded(ciks, N=20, FULLJSON=FULLJSON):
    "Multi-threaded wrapper for Yahoo Opening Prices"

    def opening_prices(cik):
        "Gets opening prices for IPO"

        ticker = FULLJSON[cik]['Company Overview']['Proposed Symbol']
        coname = FULLJSON[cik]['Company Overview']['Company Name']
        status = FULLJSON[cik]['Company Overview']['Status']
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
    #     FULLJSON[cik]['Opening Prices'] = pricedict[cik]




def spinoff_filter():
    "Greps trhough filings for spinoffs"

    spinoffs = {
        '1323885': 'ATRICURE, INC.',
        '1405082': 'TRIPLECROWN ACQUISITION CORP.',
        '1432732': 'TRIVASCULAR TECHNOLOGIES, INC.',
        '1597033': 'SABRE CORP',
        '1350031': 'EMBARQ CORP',
        '1368802': 'ATLAS ENERGY RESOURCES, LLC',
        '1506932': 'LONE PINE RESOURCES INC.',
        '1345111': 'TIM HORTONS INC.',
        '1471055': 'BANCO SANTANDER (BRASIL) S.A.',
        '1295172': 'RISKMETRICS GROUP INC',
        '1392522': 'FREESCALE SEMICONDUCTOR, LTD.',
        '1386787': 'VICTORY ACQUISITION CORP',
        '1308208': 'UNIVERSAL TRUCKLOAD SERVICES, INC.',
        '1365101': 'PRIMO WATER CORP',
        '1322734': 'ADVANCED LIFE SCIENCES HOLDINGS, INC.',
        '1365135': 'WESTERN UNION CO',
        '1434621': 'TREE.COM, INC.',
        '1434620': 'INTERVAL LEISURE GROUP, INC.',
    }

    def is_spinoff(cik):
        " Greps through S-1 filings to see if IPO is a spin-off"

        from subprocess import Popen, PIPE
        filingdir = glob.glob(os.path.join(FILEDIR, cik) + '/*')
        for filing in filingdir:
            grep_str = "egrep '( our spin-off|the spin-off will)' {}".format(filing)
            std_out = Popen(grep_str, shell=True, stdout=PIPE).stdout.read()
            if std_out:
                print("{}: {} is a spin-off".format(firmname(cik), cik))
                if DEBUG: print(std_out.decode('latin-1').replace('&nbsp;',''))
                return True
        return False

    for cik in FINALJSON:
        print("Checking if {}: {} is a spin-off{}".format(cik, firmname(cik), ' '*30), end='\r')
        if cik in spinoffs:
            FINALJSON[cik]['Metadata']['Spinoff'] = True
            continue
        else:
            FINALJSON[cik]['Metadata']['Spinoff'] = is_spinoff(cik)
    # spinoffs.update({cik:firmname(cik) for cik in ciks[1200:] if is_spinoff(cik)})








if __name__=='__main__':

    ######## NEW BAD INDUSTRIES ######################################
    bad_sic = [
        '6021', '6022', '6035', '6036', '6111', '6199', '6153',
        '6159', '6162', '6163', '6172', '6189', '6200', '6022',
        '6221', '6770', '6792', '6794', '6795', '6798', '6799',
        '8880', '8888', '9721', '9995'
        ]
    # {cik:firmname(cik) for cik in FINALJSON if FINALJSON[cik]['Metadata']['SIC'] == '6021'}
    # FINALJSON = {cik:vals for cik,vals in FINALJSON.items()
    #              if vals['Metadata']['SIC'] not in bad_sic}
    ###################################################################
