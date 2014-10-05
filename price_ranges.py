
import os
import re
import json
import glob

import requests
import arrow
import pandas as pd
import numpy as np

from lxml           import etree, html
from pprint         import pprint
from IPython        import embed
from widgets        import fix_bad_str, fix_dollars, \
                        view_filing, as_cash, write_FINALJSON


DEBUG = True
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ")
FILEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "Filings")
FINALJSON = json.loads(open(BASEDIR + '/final_json.txt').read())

aget = lambda x: arrow.get(x, 'M/D/YYYY')
conames_ciks = {cik:FINALJSON[cik]['Company Overview']['Company Name'] for cik in FINALJSON}
firmname = lambda cik: conames_ciks[cik]
get_cik = lambda firm: [x[0] for x in conames_ciks.items() if x[1].lower().startswith(firm)][0]




def parse_section(html):
    """Looks through efx_elem nodes for price range elements to parse.
    Yields sentences to parse_sentence() to check for price ranges"""

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
    "Looks through the first 3 tables for IPO price ranges."

    offer_price = re.compile(r'[Oo]ffering\s*[Pp]rice')
    common_stock = re.compile(r'[Cc]ommon\s*[Ss]tock')
    per_share = re.compile(r'(^[Pp]er [Ss]hare|Per ADS)$')
    earnings = re.compile(r'[Ee]arnings')

    for N in range(1,6):
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
                # print(s)
                yield '<Table>: Initial public offering price per share is ' + s
                break


def parse_sentence(sentence):
    "Parses sentence/paragraph for IPO price ranges and returns a price range"

    # Filter options, preferreds, convertibles
    ex_option = re.compile(r"[Ee]xercise\s*[Pp]rice")
    convertible = re.compile(r"[Cc]onvertible note[s]?")
    preferred = re.compile(r"[Cc]onvertible preferred stock")
    private = re.compile(r"([Pp]rivate placement[s]?|[Ss]eries [A-Z])")
    warrant = re.compile(r"([Ww]arrant|[Ee]xercisable)")

    # Check is IPO relevant paragraph
    is_an_ipo = re.compile(r"[Ii]nitial\s+[Pp]ublic\s+[Oo]ffering")
    common_stock = re.compile(r"common stock in the offering")
    no_public = re.compile(r"(no\s*(established)?\s*public\s*(trading)?\s*market" +
                           r"|no\s*current\s*market\s*for\s*our\s*([Cc]ommon)?\s*[Ss]tock"
                           r"|not been a public market for our [Cc]ommon\s*[Ss]tock)")

    # price extraction rules
    price_rng = re.compile(r"(\$\s{0,1}\d*[\.]?\d*\s+[Aa]nd\s+(U[\.]?S)?\s*\$\s{0,1}\d*[\.]?\d*\s[Mm][i][l]" + r"|\$\d*[\.]?\d*\s+[Aa]nd\s+(U[\.]?S)?\$\d*[\.]?\d*)")
    prices_strict = re.compile(r"(offering price (of|is) \$\d+[\.]\d+ per (share|ADS)" +
                               r"|offered at a price of \$\d+[\.]\d+ per share" +
                               r"|offering price per (share|ADS) is \$\d+[\.]\d+" +
                               r"|offering price (of the|per) ADS[s]? is (U[\.]?S)?\$\d+[\.]\d+)")
    price_types = re.compile(r"(\$\d*[\.]?\d{0,2}....|\$\d*[\.]?\d{0,2})")
    # price_rng and price_types catches $19.00 mil,
    oprice = re.compile(r"\$\s{0,1}\d*[\.]?\d{0,2}")

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





def extract_all_price_range(ciks, FINALJSON=FINALJSON):
    """Parses and extracts all price ranges from all filings for all firm ciks.
    Modifies FINALJSON dictionary in-place."""

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

        # if len(FINALJSON[cik]['Filing'][0]) > 4:
        #     print("Skipping %s %s" % (cik, coname))
        #     skipped_ciks |= {cik}
        #     continue

        if len([as_cash(s[4][0]) for s in FINALJSON[cik]['Filing'] if as_cash(s[4][0])]) > 1:
            print("Skipping %s %s" % (cik, coname))
            skipped_ciks |= {cik}
            continue


        print("\n==> Getting Price Range for: '%s' # %s" % (cik, coname))
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





def print_pricerange(s):
    if not s.isdigit():
        cik = [k for k in FINALJSON if firmname(k).lower().startswith(s.lower())][0]
    else:
        cik = s
    print("===> Filing Price Range: %s: %s <===" % (firmname(cik), cik))
    pprint([[v[2], v[1], v[-1]] for v in FINALJSON[cik]['Filing']])
    print("="*40+'\n')
# '1467858' -> GM




def testfiles(cik):
    return [x for x in glob.glob(os.path.join(FILEDIR, cik) + '/*') if 'filing.ashx?' in x]



if __name__=='__main__':

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
    cik = '0700923' # MYR Group
    cik = '1575793' # ENERGOUS CORP
    cik = '1324479' # AMERICAN COMMERCIAL LINES INC.
    cik = '1537435' # TECOGEN INC
    cik = '1379009' # GAZIT-GLOBE LTD # Israeli firm, quotes Tel Aviv prices instead of price range


    cik = '1175685' # Bladelogic
    cik = '1500435' # GoPro
    cik = '1271024' # LinkedIn

    tf = glob.glob(os.path.join(FILEDIR, cik) + '/*')
    price_range = [get_price_range(f) for f in tf]
    ciks = sorted(FINALJSON.keys())

    # Units, ADRs, etc
    # metadata[metadata['Issue Type Code'] != '0']


