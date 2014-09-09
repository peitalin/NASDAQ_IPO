
import os, time, re, string, json, glob
import dateutil.parser
import pandas as pd
import requests
import arrow

from string         import capwords
from lxml           import etree, html
from itertools      import *
from functools      import reduce
from pprint         import pprint
from IPython        import embed
from widgets        import Writer, GrepProgressbar, parallelize
from concurrent.futures import ThreadPoolExecutor

DEBUG = True
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "Filings")
FINALJSON = json.loads(open('final_json.txt').read())


def fix_bad_str(string):
    """ Formats nuisance characters. """
    if string:
        # These are various long dash characters used in the document
        for r in [r'\x97', r'\x96', r'[-]+', r'●']:
            string = re.sub(r, '-', string)
        # Other nuisance chars
        string = re.sub(r'\x95', '->', string)
        string = re.sub(r'\x93', '"', string)
        string = re.sub(r'\x94', '"', string)
        string = re.sub(r'/s/', '', string)
        string = re.sub(r'\x92', "'", string)
        string = re.sub(r'\xa0', ' ', string)
        string = re.sub(r'\s+', ' ', string)
    return string.strip()

def fix_dollars(string_list):
    """Split and strip a string, appending dollar signs where necessary"""
    new_strlist = []
    prepend_next = False
    for s in string_list:
        s = re.sub(r'^(U[\.]?S[\.]?)?\$\s*', '$', s)
        if s.strip() == '$':
            prepend_next = True
            continue
        new_str = ' '.join(e.strip() for e in s.split('\n'))
        if prepend_next == True:
            if not new_str.startswith('$'):
                new_str = '$' + new_str
            prepend_next = False

        new_strlist += [new_str]
    return new_strlist

def view_filing(filename):
    newfilename = '~/Public/' + filename.split('/')[-1] + '.html'
    os.system("cp {0} {1}".format(filename, newfilename))
    os.system("open -a Firefox {}".format(newfilename))




def parse_section(html):
    elem_types  = "[self::p or self::div]"
    sub_headers = ["efx_subject_stock_info", "",
                   "efx_the_offering",
                   "efx_registration_fee",
                   "efx_financial_data"]

    for subheader in sub_headers:
        efx_elem = html.xpath("//body/efx_form/{HEAD}/*{ELEM}".format(HEAD=subheader, ELEM=elem_types))
        if efx_elem:
            yield from (" ".join([fix_bad_str(s) for s in elem.xpath(".//text()")]) for elem in efx_elem)



def parse_table(html):
    "Looks through the first 3 tables for IPO prices"

    offer_price = re.compile(r'[Oo]ffering [Pp]rice')
    common_stock = re.compile(r'[Cc]ommon [Ss]tock')
    per_share = re.compile(r'([Pp]er [Ss]hare|Per ADS)')

    for N in range(1,4):
        # Look at first 3 efx_tables with: <div>, <table>, <center> elems
        xtable = '//body/efx_form//efx_unidentified_table[{N}]/*[self::div or self::table or self::center]//tr/descendant::*/text()'
        efx_table_text = html.xpath(xtable.format(N=N))
        efx_elems_text = fix_dollars([fix_bad_str(s) for s in efx_table_text if fix_bad_str(s)])

        next_elem_is_IPO_price = False
        for s in efx_elems_text:
            if not s:
                continue
            if offer_price.search(s) or common_stock.search(s) or per_share.search(s):
                next_elem_is_IPO_price = True
                continue
            if next_elem_is_IPO_price and s.startswith('$'):
                yield 'Initial public offering price per share is ' + s
                break



def parse_sentence(sentence):

    # Filter options, preferreds, convertibles
    ex_option = re.compile(r"[Ee]xercise\s*[Pp]rice")
    convertible = re.compile(r"[Cc]onvertible note[s]?")
    preferred = re.compile(r"[Cc]onvertible preferred stock")
    private = re.compile(r"([Pp]rivate placement[s]?|[Ss]eries [A-Z])")
    warrant = re.compile(r"([Ww]arrant|[Ee]xercisable)")

    # Check is IPO relevant paragraph
    is_an_ipo = re.compile(r"[Ii]nitial\s+[Pp]ublic\s+[Oo]ffering")
    common_stock = re.compile(r"common stock in the offering")
    no_public = re.compile(r"no\s*(established)? public market")

    # price extraction rules
    price_rng = re.compile(r"\$\d*[.]?\d*\s+[Aa]nd\s+(U[\.]?S)?\$\d*[.]?\d*")
    prices_strict = re.compile(r"(offering price (of|is) \$\d+[.]\d+ per (share|ADS)" +
                               r"|offered at a price of \$\d+[.]\d+ per share" +
                               r"|offering price per (share|ADS) is \$\d+[.]\d+" +
                               r"|offering price (of the|per) ADS[s]? is (U[\.]?S)?\$\d+[.]\d+)")
    prices = re.compile(r'\$\d*[.]\d{0,2}')

    s = sentence
    if any([ex_option.search(s), convertible.search(s), preferred.search(s), private.search(s)]):
        return None

    # if warrant.search(s):
    #     print("{} is a warrant IPO")

    if is_an_ipo.findall(s) or common_stock.findall(s) or no_public.findall(s):
        if re.findall(r'fair value', s):
            return None # 'Fair value' hypothetical prices

        if price_rng.findall(s):
            if DEBUG: print(s)
            return prices.findall(price_rng.search(s).group(0))

        if prices_strict.findall(s):
            if DEBUG: print(s)
            return prices.findall(prices_strict.search(s).group(0))
            # finds phrases of the form "offering price of $xx.xx per share"


def get_price_range(filename):

    with open(filename) as f:
        html = etree.HTML(f.read())

    # Parse text sections
    for price_range in map(parse_sentence, parse_section(html)):
        if price_range:
            return price_range

    # Parse tables
    for price_range in map(parse_sentence, parse_table(html)):
        if price_range:
            return price_range



def merge_price_range(cik, price_range):
    cik_filing = FINALJSON[cik]['Filing'].copy()
    price_filings = []
    if all([p[0] in c[3] for p,c in zip(reversed(price_range), cik_filing)]):
        for p, filing in zip(reversed(price_range), cik_filing):
            price = p[1]
            price_filings.append(filing + [price])
    elif not all([p[0] in l[3] for p,l in zip(pr, cik_filing)]):
        raise Exception("Price range IDs don't align with NASDAQ filings!!")
    return price_filings



def testfiles(cik):
    return [x for x in glob.glob(os.path.join(BASEDIR, cik) + '/*') if 'filing.ashx?' in x]

def test_cik(cik):
    tf = [x for x in glob.glob(os.path.join(BASEDIR, cik) + '/*') if 'filing.ashx?' in x]
    ids = [f[-16:] for f in tf]
    return cik, tf, list(zip(ids, [get_price_range(f) for f in tf]))







def opening_prices_list(ciks):

    prices_dict = {}
    for cik in ciks:
        prices_dict[cik] = opening_prices(cik)
    return prices_dict

    def opening_prices(cik):
        "Gets opening prices for IPO"

        ticker = FINALJSON[cik]['Company Overview']['Proposed Symbol']
        coname = FINALJSON[cik]['Company Overview']['Company Name']
        status = FINALJSON[cik]['Company Overview']['Status']
        listing_date = arrow.get(re.findall(r'\d+/\d+/\d+', status)[0], 'M/D/YYYY')
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

        return dict(zip(headers, prices))


def opening_prices_threaded(ciks, N=20):
    "Multi-threaded wrapper for Yahoo Opening Prices"
    with ThreadPoolExecutor(max_workers=N) as exec:
        json_result = exec.map(opening_prices_list, [iter(ciks)]*N)
        final_dict  = reduce(lambda d1, d2: dict(d1, **d2), json_result)
    return final_dict



def merge_CRSP_FINALJSON():
    # CRSP = pd.read_csv("CRSP.csv", dtype=object)
    # pricedict = opening_prices_threaded(ciks, N=20)
    # for cik in pricedict:
    #     FINALJSON[cik]['Opening Prices'] = pricedict[cik]

    CRSP2 = pd.read_csv("data/CRSP2_openingprices.csv", dtype=object)
    CRSP2.set_index("CIK", inplace=True)

    for cik in CRSP2.index:
        if cik not in FINALJSON.keys():
            continue

        FINALJSON[cik]['Metadata']['CUSIP'] = CRSP2.loc[cik, 'CUSIP']
        FINALJSON[cik]['Metadata']['GVKEY'] = [str(s) for s in CRSP2.loc[cik, ['GVKEY', 'IID']].tolist()]
        FINALJSON[cik]['Metadata']['NAICS'] = CRSP2.loc[cik, 'NAICS']

        FINALJSON[cik]['Opening Prices'] = {}
        price_dict = {'Volume': CRSP2.loc[cik, 'Volume'],
                    'Close': CRSP2.loc[cik, 'Close'],
                    'High': CRSP2.loc[cik, 'High'],
                    'Low': CRSP2.loc[cik, 'Low'],
                    'Open': CRSP2.loc[cik, 'Open'],
                    'Date': CRSP2.loc[cik, 'Date']}
        FINALJSON[cik]['Opening Prices'] = price_dict






if __name__=='__main__':

    ######## NEW BAD INDUSTRIES #############
    # bad_sic = [6035, 6036, 6099, 6111, 6153, 6159, 6162, 6163, 6172, 6189, 6200, 6022, 6221, 6770, 6792, 6794, 6795, 6798, 6799, 8880, 8888, 9721, 9995]
    # FINALJSON = {cik:vals for cik,vals in FINALJSON.items() if int(vals['Metadata']['SIC code']) not in bad_sic}
    # with open('final_json.txt', 'w') as f:
    #     f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))

    cik = '1326801' # Facebook
    cik = '1594109' # grubhub           -> 2 upwards price revisions
    cik = '1117733' # cafepress         -> 1 upwards price revision
    cik = '1167896' # nexttest systems  -> 1 downwards price revision
    cik = '1168197' # liposcience       -> 2 downwards price revision
    cik = '1169561' # commvault systems -> 1 upwards price revision
    cik = '1169652' # channeladvisor    -> 1 upwards price revision
    cik = '1296391' # tengion inc       -> 1 large downwards
    cik = '1499934' # country-style
    cik = '1468174' # hyatt hotels
    cik = '1485538' # Ossen innovation -> good F-1 filing example
    # problem firms
    cik = '1376972' # home inns and hotels mgmt, F-1 foreign listings
    cik = '1407031' # golden pond (warrant filing)
    cik = '1544856' # CENCOSUD SA'
    cik = '1175685' # Bladelogic
    cik = '1500435' # GoPro

    tf = glob.glob(os.path.join(BASEDIR, cik) + '/*')
    pr = [get_price_range(f) for f in tf]

    ciks = iter(FINALJSON.keys())

    company_overview = pd.DataFrame([FINALJSON[cik]['Company Overview'] for cik in FINALJSON.keys()], FINALJSON.keys())
    financials = pd.DataFrame([FINALJSON[cik]['Financials'] for cik in FINALJSON.keys()], FINALJSON.keys())
    experts = pd.DataFrame([FINALJSON[cik]['Experts'] for cik in FINALJSON.keys()], FINALJSON.keys())


    CRSP2 = pd.read_csv("data/CRSP2_openingprices.csv", dtype=object)
    CRSP2.set_index("CIK")
    firms = set(FINALJSON.keys()) # 1541
    crspfirms = set(CRSP2.index) # 1521
    missingfirms = firms - crspfirms # 34




    # X1) Get S-1 filings from NASDAQ
    # X2) GET SIC codes from edgar / rawjson.txt
    # X3) Filter bad SIC codes
    # X4) Match CRSP/Yahoo Firms with NASDAQ firms for opening/close prices
    # 4) Get WRDS access and US CRSP dataset for stock prices.
    # 5) Fix Gtrends names, start scraping attention
    # 8) Get partial price adjustments and put in "Filings"






def pricerng_all(ciks):

    ciks = iter(FINALJSON.keys())
    missing_pr = []

    for cik in ciks:
        coname = FINALJSON[cik]['Company Overview']['Company Name']
        if len(FINALJSON[cik]['Filing'][0]) > 4:
            print("Skipping {} {}".format(cik, coname))
            continue

        print('Getting Price Range for: {} {}'.format(cik, coname))
        cik, tf, pr = test_cik(next(ciks))
        list_pr = merge_price_range(cik, pr)

        if any([x[4] for x in list_pr]):
            print("Did not find price range for {}".format(coname))
            missing_pr.append(cik)

        FINALJSON[cik]['Filing'] = list_pr
        print('\n')





