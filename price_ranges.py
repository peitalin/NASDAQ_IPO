
import os, time, re, string, json, glob
import dateutil.parser
import pandas as pd
import requests
import arrow
import numpy as np

from string         import capwords
from lxml           import etree, html
from itertools      import *
from functools      import reduce
from pprint         import pprint
from IPython        import embed
from widgets        import fix_bad_str, fix_dollars, view_filing, safari, firefox
from concurrent.futures import ThreadPoolExecutor

DEBUG = True
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "Filings")
FINALJSON = json.loads(open('final_json.txt').read())



def parse_section(html):

    sub_headers = ["efx_subject_stock_info", "",
                   "efx_the_offering",
                   "efx_registration_fee",
                   "efx_financial_data"]

    for subheader in sub_headers:
        if subheader:
            elem_types = "[self::p or self::div]"
        else:
            Nth_elem = int(len(html.xpath("//body/efx_form//*")) / 4)
            elem_types = "[position() < {} and (self::p or self::div)]".format(Nth_elem)

        efx_path = "//body/efx_form/{HEAD}/*{ELEM}"
        efx_elem = html.xpath(efx_path.format(HEAD=subheader, ELEM=elem_types))
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
                yield '<Table>: Initial public offering price per share is ' + s
                break


def parse_sentence(sentence):

    def surround_words(sentence, re_pattern, n):
        """retrieves n words either side of price"""
        word = r"\W*([\w]+)"
        oprice = r"\$\d*[\.]?\d{0,2}"
        regexp = r'{}\W*{}{}'.format(word*n, re_pattern.pattern, word*n)
        return re.search(regexp, sentence).group()

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
    price_rng = re.compile(r"(\$\d*[\.]?\d*\s+[Aa]nd\s+(U[\.]?S)?\$\d*[\.]?\d*\s[Mm][i][l]" + r"|\$\d*[\.]?\d*\s+[Aa]nd\s+(U[\.]?S)?\$\d*[\.]?\d*)")
    prices_strict = re.compile(r"(offering price (of|is) \$\d+[\.]\d+ per (share|ADS)" +
                               r"|offered at a price of \$\d+[\.]\d+ per share" +
                               r"|offering price per (share|ADS) is \$\d+[\.]\d+" +
                               r"|offering price (of the|per) ADS[s]? is (U[\.]?S)?\$\d+[\.]\d+)")
    price_types = re.compile(r"(\$\d*[\.]?\d{0,2}....|\$\d*[\.]?\d{0,2})")
    # price_rng and price_types catches $19.00 mil,
    oprice = re.compile(r"\$\d*[\.]?\d{0,2}")

    s = sentence
    if any([ex_option.search(s),
            convertible.search(s),
            preferred.search(s),
            private.search(s)]):
        return None

    if is_an_ipo.findall(s) or common_stock.findall(s) or no_public.findall(s):
        if re.findall(r'fair value', s):
            return None # 'Fair value' hypothetical prices

        if price_rng.search(s):
            if DEBUG: print('\n', s)
            # if DEBUG: print(surround_words(s, oprice, 4))
            offer_prices = price_types.findall(price_rng.search(s).group())
            if any(['mil' in s for s in offer_prices]):
                return None
            else:
                return sum([oprice.findall(s) for s in offer_prices], [])

        if prices_strict.search(s):
            if DEBUG: print('\n', s)
            offer_prices = price_types.findall(prices_strict.search(s).group())
            if any(['mil' in s for s in offer_prices]):
                return None
            else:
                return sum([oprice.findall(s) for s in offer_prices], [])
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



def as_cash(string):
    cash_str = string.replace("$", '').replace(",",'')
    return np.float64(cash_str) if cash_str else None


def testfiles(cik):
    return [x for x in glob.glob(os.path.join(BASEDIR, cik) + '/*') if 'filing.ashx?' in x]

def fdir_pricerange(cik):
    filingdir = glob.glob(os.path.join(BASEDIR, cik) + '/*')
    tf = [x for x in filingdir if 'filing.ashx?' in x]
    ids = [f[-16:] for f in tf]
    return cik, tf, list(zip(ids, [get_price_range(f) for f in tf]))



# def is_spin_of():



def merge_price_range(cik, price_range):

    cik_filing = [c[:4] for c in FINALJSON[cik]['Filing']]
    filing_ids = [c[3] for c in cik_filing]
    # pr: price range
    pr_ids, prices = list(zip(*reversed(price_range)))
    # Check price ranges align with nasdaq filings

    pr_filings_aligned = all([p in f for p,f in zip(pr_ids, filing_ids)])

    if pr_filings_aligned:
        new_pr_filings = [f+[p] for f,p in zip(cik_filing, prices)]
        flatten_prices = sum([x for x in prices if x],[])
        flat_prices = [as_cash(d) for d in flatten_prices if as_cash(d)]
        if flat_prices != []:
            pmin, pmax = min(flat_prices), max(flat_prices)
        else:
            pmin, pmax = None, None
        return new_pr_filings, pmax, pmin
    else:
        print(cik, cik_filing[0])
        raise Exception("Price range IDs don't align with NASDAQ filings!!")



def pricerng_all(ciks):

    def fdir_pricerange(cik):
        filingdir = glob.glob(os.path.join(BASEDIR, cik) + '/*')
        tf = [x for x in filingdir if 'filing.ashx?' in x]
        ids = [f[-16:] for f in tf]
        return cik, tf, list(zip(ids, [get_price_range(f) for f in tf]))

    missing_ciks = []
    abnormal_ciks = []
    DEBUG = False

    ciks = sorted(list(set(FINALJSON.keys())))[50:100]
    for cik in ciks:
        coname = FINALJSON[cik]['Company Overview']['Company Name']
        # if len(FINALJSON[cik]['Filing'][0]) > 4:
        #     print("Skipping {} {}".format(cik, coname))
        #     continue
        print('\n==> Getting Price Range for: {} {}'.format(cik, coname))
        cik, tf, price_range = fdir_pricerange(cik)
        if not tf:
            print("Missing filings for {}".format(coname))
            missing_ciks.append(cik)
            continue

        list_pr, pmax, pmin = merge_price_range(cik, price_range)
        if not pmax:
            print("Did not find price range for {}".format(coname))
            missing_ciks.append(cik)

        elif (pmax - pmin) > 10:
            print("!!! {} => price range seems large: {}~{}".format(coname, pmax, pmin))
            abnormal_ciks.append(cik)

        FINALJSON[cik]['Filing'] = list_pr
        pprint(price_range)
        print('=== Updated price range for {} ===\n'.format(cik))








def firmname(cik):
    return FINALJSON[cik]['Company Overview']['Company Name']

def is_CUSIP_first_offer(cik):
    """Checks last -3:-1 digits to see whether the offer is truly the 1st equity offering. E.g CUSIP: '470359100' gives '10' which is then first equity offer."""
    CUSIP = FINALJSON[cik]['Metadata']['CUSIP']
    print("{} => CUSIP: {}".format(firmname(cik), CUSIP))
    return CUSIP[-3:-1] == '10'




if __name__=='__main__':

    ######## NEW BAD INDUSTRIES ######################################
    # bad_sic = [6035, 6036, 6099, 6111, 6153, 6159, 6162, 6163, 6172,
    #            6189, 6200, 6022, 6221, 6770, 6792, 6794, 6795, 6798,
    #            6799, 8880, 8888, 9721, 9995]
    # FINALJSON = {cik:vals for cik,vals in FINALJSON.items()
    #              if int(vals['Metadata']['SIC code']) not in bad_sic}
    ###################################################################

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
    cik = '1271024' # LinkedIn
    cik = '1500435' # GoPro
    cik = '1345016' # Yelp!
    cik = '1350031' # Embarq -> all none

    # tf = glob.glob(os.path.join(BASEDIR, cik) + '/*')
    # pr = [get_price_range(f) for f in tf]

    # ciks = iter(FINALJSON.keys())

    # company_overview = pd.DataFrame([FINALJSON[cik]['Company Overview'] for cik in FINALJSON.keys()], FINALJSON.keys())
    # financials = pd.DataFrame([FINALJSON[cik]['Financials'] for cik in FINALJSON.keys()], FINALJSON.keys())
    # experts = pd.DataFrame([FINALJSON[cik]['Experts'] for cik in FINALJSON.keys()], FINALJSON.keys())







