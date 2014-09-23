
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
from subprocess     import Popen, PIPE
from IPython        import embed
from widgets        import fix_bad_str, fix_dollars, view_filing
from concurrent.futures import ThreadPoolExecutor

DEBUG = True
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ")
FILEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "Filings")
FINALJSON = json.loads(open(BASEDIR + '/final_json.txt').read())


def as_cash(string):
    if '$' not in string:
        return None
    string = string.replace('$','').replace(',','')
    return float(string) if string else None


def parse_section(html):

    sub_headers = ["efx_subject_stock_info", "",
                   "efx_the_offering",
                   "efx_registration_fee",
                   "efx_financial_data"]

    for subheader in sub_headers:
        if subheader:
            elem_types = "[self::p or self::div]"
        else:
            Nth_elem = int(len(html.xpath("//body/efx_form//*")) / 10)
            # scan the first 1/10 of all general efx_from elems
            elem_types = "[position() < {} and (self::p or self::div)]".format(Nth_elem)

        efx_path = "//body/efx_form/{HEAD}/*{ELEM}"
        efx_elem = html.xpath(efx_path.format(HEAD=subheader, ELEM=elem_types))
        if efx_elem:
            yield from (" ".join([fix_bad_str(s) for s in elem.xpath(".//text()")]) for elem in efx_elem)


def parse_table(html):
    "Looks through the first 3 tables for IPO prices"

    offer_price = re.compile(r'[Oo]ffering [Pp]rice')
    common_stock = re.compile(r'[Cc]ommon [Ss]tock')
    per_share = re.compile(r'(^[Pp]er [Ss]hare|Per ADS)$')
    earnings = re.compile(r'[Ee]arnings')

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
                if not earnings.search(s):
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
            # if DEBUG: print(surround_words(s, oprice, 4))
            offer_prices = price_types.findall(price_rng.search(s).group())
            if any(['mil' in x for x in offer_prices]):
                return None
            else:
                oprices = sum([oprice.findall(x) for x in offer_prices], [])
                test_prices = [as_cash(x) for x in oprices if as_cash(x)]

                if test_prices:
                    if max(test_prices) > 150:
                        raise(Exception("Massive Offer Price, above 150!"))

                return sum([oprice.findall(x) for x in offer_prices], [])

        if prices_strict.search(s):
            if DEBUG: print('\n', s)
            offer_prices = price_types.findall(prices_strict.search(s).group())
            if any(['mil' in x for x in offer_prices]):
                return None
            else:
                return sum([oprice.findall(x) for x in offer_prices], [])
                # finds phrases of the form "offering price of $xx.xx per share"


def get_price_range(filename):

    with open(filename, encoding='latin-1') as f:
        html = etree.HTML(f.read())

    # Parse text sections
    for price_range in map(parse_sentence, parse_section(html)):
        if price_range:
            return price_range

    # Parse tables
    for price_range in map(parse_sentence, parse_table(html)):
        if price_range:
            return price_range

    return ["NA"]



def testfiles(cik):
    return [x for x in glob.glob(os.path.join(FILEDIR, cik) + '/*') if 'filing.ashx?' in x]



def extract_all_price_range(ciks, FINALJSON=FINALJSON):
    "Parses and extracts all price ranges from all filings for all firm ciks."

    def fdir_pricerange(cik):
        "For a given cik ID, gets all IPO filings from filing directory."
        filingdir = glob.glob(os.path.join(FILEDIR, cik) + '/*')
        tf = [x for x in filingdir if 'filing.ashx?' in x and x[-6:].isdigit()]
        ids = [f[-28:] for f in tf]
        return cik, tf, list(zip(ids, [get_price_range(f) for f in tf]))

    def merge_price_range(cik, price_range):
        "Grabs 'Filing' dict from FINALJSON, updates with price ranges"
        filings = [c[:4] for c in FINALJSON[cik]['Filing']]
        # c[:4] so list doesn't duplicate price ranges
        for i, filing in enumerate(filings):
            f = filing[3] # filing_name
            if f in dict(price_range):
                filings[i].append(dict(price_range)[f])
            else:
                print(filing)
                raise Exception("%s: price range IDs don't match with filings!" % cik)
        return filings

    def absolute_price_range(price_range):
        "Gets the min and max price ranges from all price ranges. For checking."
        all_prices = sum([p[1] for p in price_range if p[1]], [])
        all_prices = sorted(as_cash(d) for d in all_prices if as_cash(d))
        pmin = min(all_prices) if all_prices else None
        pmax = max(all_prices) if all_prices else None
        return pmax, pmin


    missing_ciks = set()
    skipped_ciks = set()
    abnormal_ciks = set()
    DEBUG = False

    ciks = sorted(list(set(FINALJSON.keys())))
    for i, cik in enumerate(ciks):
        coname = FINALJSON[cik]['Company Overview']['Company Name']
        if len(FINALJSON[cik]['Filing'][0]) > 4:
            print("Skipping %s %s" % (cik, coname))
            skipped_ciks |= {cik}
            continue


        print('\n==> Getting Price Range for: %s %s' % (cik, coname))
        cik, tf, price_range = fdir_pricerange(cik)
        if not tf:
            print("Missing filings for {}".format(coname))
            missing_ciks |= {cik}
            continue

        list_pr = merge_price_range(cik, price_range)
        pmax, pmin = absolute_price_range(price_range)
        if not pmax:
            print("Did not find price range for %s" % coname)
            missing_ciks |= {cik}

        elif (pmax - pmin) > 10:
            print("!! %s => price range is way big: %s~%s" % (coname, pmax, pmin))
            abnormal_ciks |= {cik}

        FINALJSON[cik]['Filing'] = list_pr
        pprint(price_range)
        print('=== Updated price range for %s ===\n' % cik)


    # abnormal_ciks = {'1024305', '1062781', '1117106', '1161448', '1208208', '1271024', '1274494', '1307954', '1311596', '1326732', '1335793', '1347557', '1361983', '1365742', '1388319', '1395213', '1401257', '1411158', '1419945', '1442596', '1467858', '1474952', '1477156', '1477641'}




def print_pricerange(s):
    if not s.isdigit():
        cik = [k for k in FINALJSON if firmname(k).lower().startswith(s.lower())][0]
    else:
        cik = s
    print("===> Filing Price Range: %s: %s <===" % (firmname(cik), cik))
    pprint([[v[2], v[1], v[-1]] for v in FINALJSON[cik]['Filing']])
    print("="*40+'\n')
# '1467858' -> GM


def firmname(cik):
    return FINALJSON[cik]['Company Overview']['Company Name']





if __name__=='__main__':

    ######## NEW BAD INDUSTRIES ######################################
    # bad_sic = [
    #     '6021', '6022', '6035', '6036', '6111', '6199', '6153',
    #     '6159', '6162', '6163', '6172', '6189', '6200', '6022',
    #     '6221', '6770', '6792', '6794', '6795', '6798', '6799',
    #     '8880', '8888', '9721', '9995'
    #     ]
    # {cik:firmname(cik) for cik in FINALJSON if FINALJSON[cik]['Metadata']['SIC'] == '6021'}
    # FINALJSON = {cik:vals for cik,vals in FINALJSON.items()
    #              if vals['Metadata']['SIC'] not in bad_sic}
    ###################################################################

    # with open('final_json.txt', 'w') as f:
    #     f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))

    cik = '1326801' # Facebook
    cik = '1318605' # Tesla Motors
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
    cik = '1315657' # 'XOOM CORP'
    cik = '1350031' # Embarq -> all none

    tf = glob.glob(os.path.join(FILEDIR, cik) + '/*')
    price_range = [get_price_range(f) for f in tf]

    # ciks = iter(FINALJSON.keys())
    ciks = sorted(FINALJSON.keys())

    company     = pd.DataFrame([FINALJSON[cik]['Company Overview']  for cik in ciks], ciks)
    financials  = pd.DataFrame([FINALJSON[cik]['Financials']        for cik in ciks], ciks)
    experts     = pd.DataFrame([FINALJSON[cik]['Experts']           for cik in ciks], ciks)
    metadata    = pd.DataFrame([FINALJSON[cik]['Metadata']          for cik in ciks], ciks)
    filings     = pd.DataFrame([FINALJSON[cik]['Filing']            for cik in ciks], ciks)
    open_prices = pd.DataFrame([FINALJSON[cik]['Opening Prices']    for cik in ciks], ciks)


    # Units, ADRs, etc
    # metadata[metadata['Issue Type Code']!='0']




    ## get_s1_filings
    ## get price_ranges for new firms
    ## reconstuct dataframe, keeping BAD_SIC and units, ADRs etc.
    ## finally after getting new DF, save full_df and slowly remove
    ## bad SIC codes, and units/ADRs



def is_spinoff(cik):
    " Greps through S-1 filings to see if IPO is a spin-off"

    from subprocess import Popen, PIPE
    filingdir = glob.glob(os.path.join(FILEDIR, cik) + '/*')
    for filing in filingdir[-3:]:
        grep_str = "egrep '( our spin-off|the spin-off will)' {}".format(filing)
        std_out = Popen(grep_str, shell=True, stdout=PIPE).stdout.read()
        if std_out:
            print("{}: {} is a spin-off".format(firmname(cik), cik))
            if DEBUG: print(std_out.decode('latin-1').replace('&nbsp;',''))
            return True
    return False



def spinoff_filter():

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
        '1322734': 'ADVANCED LIFE SCIENCES HOLDINGS, INC.'
    }

    for cik in FINALJSON:
        if cik in spinoffs:
            FINALJSON[cik]['Metadata']['Spinoff'] = True
            continue
        else:
            FINALJSON[cik]['Metadata']['Spinoff'] = is_spinoff(cik)


    # spinoffs.update({cik:firmname(cik) for cik in ciks[1200:] if is_spinoff(cik)})

