
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
FILEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "filings")
FINALJSON = json.loads(open(BASEDIR + '/final_json.txt').read())
FULLJSON = json.loads(open(BASEDIR + '/full_json.txt').read())

aget = lambda x: arrow.get(x, 'M/D/YYYY')
conames_ciks = {cik:FULLJSON[cik]['Company Overview']['Company Name'] for cik in FULLJSON}
firmname = lambda cik: conames_ciks[cik]
get_cik = lambda firm: [x[0] for x in conames_ciks.items() if x[1].lower().startswith(firm)][0]



def parse_section(html):
    """Looks through efx_elem nodes for price range elements to parse.
    Yields sentences to parse_sentence() to check for price ranges"""

    sub_headers = ["efx_subject_stock_info", "",
                   "efx_the_offering",
                   "efx_registration_fee",
                   "efx_proceed_use",
                   "efx_financial_data"]

    for subheader in sub_headers:
        if subheader:
            elem_types = "[self::p or self::div or self::dl]"
            efx_path = "//body/efx_form/descendant::{HEAD}/*{ELEM}"
        else:
            stop_index = int(len(html.xpath("//body/efx_form//*")) / 8)
            # scan the first quarter of all general efx_from elems
            elem_types = "[position() < {} and (self::p or self::div or self::dl)]".format(stop_index)
            efx_path = "//body/efx_form/{HEAD}/*{ELEM}"

        efx_elem = html.xpath(efx_path.format(HEAD=subheader, ELEM=elem_types))
        if efx_elem:
            yield from (" ".join([fix_bad_str(s) for s in elem.xpath(".//text()")]) for elem in efx_elem)


def parse_table(html):
    "Looks through the first 3 tables for IPO price ranges."

    offer_price = re.compile(r'[Oo]ffering\s*[Pp]rice')
    common_stock = re.compile(r'[Cc]ommon\s*[Ss]tock')
    initial_price = re.compile(r'([Ii]nitial [Pp]rice to [Pp]ublic|[Pp]rice to [Pp]ublic)')
    per_share = re.compile(r'(^[Pp]er [Ss]hare|^[Pp]er ADS|^[Pp]er [Cc]ommon [Uu]nit|^[Pp]er [Uu]nit|^[Pp]er [Cc]ommon [Ss]hare|^Per [Oo]rdinary [Ss]hare)')
    midpoint = re.compile(r'[Mm]idpoint of the price\s*range')
    ipo_table_identifiers = [offer_price, common_stock, initial_price, per_share, midpoint]

    # negative search
    earnings = re.compile(r'[Ee]arnings')
    warrants = re.compile(r'[Ww]arrant')
    balance_sheet_headers = ['cash', 'deferred', 'tax', 'assets', 'depreciation', 'net income', 'liabilities']


    # Look at first 5 efx_unidentified_tables with: <div>, <table>, <center> elems
    efx_unidentified_table = '//body/efx_form/descendant::efx_unidentified_table[{N}]' + \
                             '/descendant::tr/descendant::*/text()'
    # poor html formatting: tables outside of efx_unidentified_table
    efx_form_table = "//body/efx_form/table[{N}]//descendant::*/text()"
    efx_registration = "//body/descendant::efx_registration_fee/descendant::efx_unidentified_table/descendant::tr/descendant::*/text()"
    efx_distribution_plan = '//efx_form/descendant::efx_distribution_plan/descendant::table/descendant::*/text()'

    table_xpaths = [efx_unidentified_table.format(N=N) for N in range(8)] + \
                   [efx_registration] + \
                   [efx_form_table.format(N=N) for N in range(5)]


    for N, xpath_table in enumerate(table_xpaths):

        # efx_table_text = html.xpath(xpath_table)
        efx_table_text = html.xpath(table_xpaths[N])
        efx_elems_text = [fix_bad_str(s) for s in efx_table_text if fix_bad_str(s)]


        for i,x in enumerate(efx_elems_text):
            if any(x.lower().startswith(b) for b in balance_sheet_headers):
                # truncate list when encountering balance sheet items
                efx_elems_text = efx_elems_text[:i]
                break

        if not any(f==s=='$' for f,s in zip(efx_elems_text, efx_elems_text[1:])):
            # checks tables for blank $, otherwise appends $ with next number.
            efx_elems_text = fix_dollars(efx_elems_text)

        next_elem_is_IPO_price = False
        counter = 0
        for s in efx_elems_text:

            if counter > 5:
                if DEBUG: print('break next_elem_is_IPO_price. {}'.format(s))
                next_elem_is_IPO_price = False
                counter = 0

            if earnings.search(s) or warrants.search(s):
                next_elem_is_IPO_price = False
                continue

            if any(id.search(s) for id in ipo_table_identifiers):
                if not earnings.search(s):
                    next_elem_is_IPO_price = True
                    counter = 0
                    continue

            if next_elem_is_IPO_price:
                counter += 1

            if re.search(r"\$[\d]{3}[\.]?\d*", s) or re.search(r"\$[\d,]*,\d{3}", s) or 'par value' in s:
                continue

            elif next_elem_is_IPO_price and (re.search(r"\$\d{0,2}[\.]?\d{0,3}", s) or s.strip()=='$'):
                if '.' in s and re.search(r'\d[05]$', s):
                    if DEBUG: print('<Table {}>: IPO price per share is {}'.format(N, s))
                    yield '<Table {}>: Initial public offering price is {} per share'.format(N, s)
                    break
                elif '.' not in s and re.search(r'\$\d\d', s):
                    if DEBUG: print('<Table {}>: IPO price per share is {}'.format(N, s))
                    yield '<Table {}>: Initial public offering price is {} per share'.format(N, s)
                    break




def parse_sentence(sentence):
    "Parses sentence/paragraph for IPO price ranges and returns a price range"

    # Filter options, preferreds, convertibles
    convertible = re.compile(r"[Cc]onvertible note[s]?")
    preferred = re.compile(r"[Cc]onvertible preferred stock")
    private = re.compile(r"([Pp]rivate placement[s]?)")
    warrant = re.compile(r"([Ww]arrant|[Ee]xercisable)")
    dividends = re.compile(r"[Pp]ermitted [Dd]ividend[s]?")
    restricted_stock = re.compile(r"[Rr]estricted [Ss]tock")
    option_filters = [convertible, preferred, private, warrant, dividends, restricted_stock]

    # Check if paragraph is IPO relevant
    is_an_ipo = re.compile(r"([Ii]nitial\s+[Pp]ublic\s+[Oo]ffering|public offering price)")
    applied_listing = re.compile(r"[Ww]e ([\w]+\s)*list ([\w]+\s)*shares ([\w]+\s)*[Ee]xchange")
    anticipate = re.compile(r"(we anticipate that ([\w]+\s)*shareholders ([\w]+\s)*sell their shares" +
                            r"|we anticipate ([\w]+\s)*price will be (U[\.]?S)?\$\s*\d+[\.]\d*)")
    offer_common = re.compile(r"[Oo]ffer(ing)? shares\s*(of)?\s*common\s*stock")
    common_stock = re.compile(r"(common stock in the offering" +
                              r"|offering price ([\w]+\s)*common shares ([\w]+\s)*selling shareholders)")
    no_public = re.compile(r"([Nn]o (established)?\s*public\s*(trading)?\s*market" +
                           r"|[Nn]o current\s*market\s*for\s*our\s*([Cc]ommon)?\s*[Ss]tock" +
                           r"|[Nn]o public market currently exists for our" +
                           r"|[Nn]ot been a public market for our [Cc]ommon\s*[Ss]tock)")
    ipo_identifiers = [is_an_ipo, applied_listing, anticipate, offer_common, common_stock, no_public]

    # price extraction rules
    price_rng = re.compile(
        r"(\$\s*\d*[\.]?\d{0,2}\s[Aa]nd\s(U[\.]?S)?\s*\$\s*\d{0,2}[\.]?\d{0,2}\s[Mm][i][l]" + \
        r"|\$\s*\d*[\.]?\d{0,2}\s[Aa]nd\s(U[\.]?S)?\s*\$\s*\d{0,2}[\.]?\d{0,2})")

    price_rng2 = re.compile(r"(price is between\s*and\s*per (common)?\s*(share|unit)" +
                            r"|be between\s*[\$]?\s*and\s*[\$]?\s*(per)?\s*(common)?\s*(share|unit)" +
                            r"|offering price will be \$\[\s*.*\s*\] (common)?\s*(share|unit)" +
                            r"|offering price of the shares will be ([\w]+\s)*\$\s*\d*[\.]?\d{0,2}\s*(-|to|~)\s*([\w]+\s)*\$\s*\d*[\.]?\d{0,2} per" +
                            r"|be between\s*\$\s*\[\s*.*\s*\]\s*and\s*\$\s*\[\s*.*\s*\]\s*)")

    prices_strict = re.compile(r"(offering price (of|is) \$\s*\d+[\.]\d* per (share|ADS)" +
                               r"|offered\s*(for sale)?\s*at a price of \$\s*\d+[\.]\d* per share" +
                               r"|offering price per (share|ADS) is \$\s*\d+[\.]\d*" +
                               r"|offering price (of the|per) ADS[s]? is (U[\.]?S)?\$\s*\d+[\.]\d*" +
                               r"|offer(ing)? price ([\w]+\s)*\$\s*\d*[\.]?\d{0,2} per (common)?\s*(share|ADS|unit)" +
                               r"|[Ww]e (expect|anticipate) ([\w]+\s)*price (will|to) be\s*(U[\.]?S)?\$\s*\d+[\.]\d{0,2}" +
                               r"|[Ii]nitial public offering price of\s*(U[\.]?S)?\$\s*\d+[\.]\d{0,2}per share" +
                               r"|offering price to be (U[\.]?S)?\$\s*\d+[\.]\d* per share)")
    price_types = re.compile(r"(\$\s*\d*[\.]?\d{0,2}\s[MmBb]il|\$\s*\d*[\.]?\d{0,2})")
    # price_rng and price_types catches $19.00 mil
    oprice = re.compile(r"\$\s*\d*[\.]?\d{0,2}")

    s = sentence
    if any(x.search(s) for x in option_filters):
        if no_public.search(s):
            split_from = no_public.search(s).span()[0]
            s = s[split_from:]
        elif anticipate.search(s):
            split_from = anticipate.search(s).span()[0]
            s = s[split_from:]
        elif price_rng2.search(s):
            s = s
        else:
            return None
        # Double check options/warrants after splitting sentence in half
        if len([x.search(s) for x in option_filters if x.search(s)]) > 1:
            return None

    if any(x.search(s) for x in ipo_identifiers):
        # if DEBUG: print(s)

        if re.search(r'fair value', s) or re.search(r'net (tangible)?\s*book value', s):
            return None # 'Fair value' hypothetical prices

        if price_rng.search(s):
            offer_prices = price_types.findall(price_rng.search(s).group())
            if any('mil' in x for x in offer_prices):
                return None
            else:
                offer_prices = [re.sub(r'[\.]$', '', oprice.search(x).group().replace(' ','')) for x in offer_prices]
                if DEBUG: print(offer_prices)
                test_prices = [as_cash(x) for x in offer_prices if as_cash(x)]

                if test_prices:
                    if max(test_prices) > 150:
                        print(s)
                        raise(Exception("Massive Offer Price, above 150!"))

                return offer_prices

        if price_rng2.search(s):
            if DEBUG: print(s)
            return ['$', '$']

        if prices_strict.search(s):
            if DEBUG: print('\nprices_strict =>', s)
            offer_prices = price_types.findall(prices_strict.search(s).group())
            if any('mil' in x for x in offer_prices):
                return None
            else:
                return [oprice.search(x).group().replace(' ','') for x in offer_prices]
                # finds phrases of the form "offering price of $xx.xx per share"








def rf(filename):
    with open(filename, encoding='latin-1') as f:
        return etree.HTML(f.read())


def get_price_range(filename):
    "Reads S-1 filings and parses for price ranges with parse_section() and parse_table() functions."

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
    done_ciks = []


    ciks = sorted(list(set(FINALJSON.keys())))


    for i, cik in enumerate(ciks):
        if cik in done_ciks:
            continue

        coname = FINALJSON[cik]['Company Overview']['Company Name']

        # if len(FINALJSON[cik]['Filing'][0]) > 4:
        #     print("Skipping %s %s" % (cik, coname))
        #     skipped_ciks |= {cik}
        #     continue

        if len([as_cash(s[4][0]) for s in FINALJSON[cik]['Filing'] if as_cash(s[4][0])]) > 2:
            print("Skipping %s %s" % (cik, coname))
            skipped_ciks |= {cik}
            continue

        # if not [s[4] for s in FINALJSON[cik]['Filing'] if s[4][0]=='NA']:
        #     continue


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
        print_pricerange(cik)
        print('=== Updated price range for %s ===\n' % cik)

        done_ciks.append(cik)







def print_pricerange(s):
    "arg s: either firmname or cik. Returns price-ranges"

    if not isinstance(s, str):
        s = str(s)
    if not s.isdigit():
        cik = [k for k in FINALJSON if firmname(k).lower().startswith(s.lower())][0]
    else:
        cik = s

    filings = FINALJSON[cik]['Filing']

    print("\n{A}> Filing Price Range: {B}: {C} <{A}".format(A='='*22, B=firmname(cik), C=cik))
    print("Date\t\tFormtype\tFile\t\t\t\tPrice Range")
    for v in filings:
        filing_ashx = v[-2]
        price_range = v[-1][0]
        if price_range in ['NA', 'NAN']:
            NA_filepath = os.path.join(BASEDIR, 'filings', cik, filing_ashx)
            filesize = os.path.getsize(NA_filepath)
            print("{}\t{}\t\t{}\t{}\t<= {:,} kbs".format(v[2], v[1], v[3], v[-1], round(filesize/1000)))
        else:
            print("{}\t{}\t\t{}\t{}".format(v[2], v[1], v[3], v[-1]))
    print("="*90+'\n')
# '1467858' -> GM






def testfiles(cik):
    return [x for x in glob.glob(os.path.join(FILEDIR, cik) + '/*') if 'filing.ashx?' in x]




if __name__=='__main__':

    # with open('final_json.txt', 'w') as f:
    #     f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))

    # with open('full_json.txt', 'w') as f:
    #     f.write(json.dumps(FULLJSON, indent=4, sort_keys=True))


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


    # Price less than $4
    # cik = '1349892' # VALUERICH INC
    # cik = '1361916' # ASIA TIME INC


    cik = '1175685' # Bladelogic
    cik = '1500435' # GoPro
    cik = '1271024' # LinkedIn

    ciks = sorted(FINALJSON.keys())


    # Units, ADRs, etc
    # metadata[metadata['Issue Type Code'] != '0']
    # FULLJSON ciks
    # badciks = ['0860413', '1070336', '1127393', '1131312', '1287668', '1290059', '1302176', '1302324', '1303942', '1308106', '1310313', '1332174', '1336249', '1337068', '1340282', '1347426', '1349892', '1353691', '1354730', '1361916', '1370433', '1370946', '1376227', '1376556', '1378239', '1379606', '1381668', '1382230', '1399521', '1401573', '1402902', '1410402', '1412203', '1434620', '1434621', '1492915', '1507385']


    ciks = ['1071625', '1276187', '1334814', '1372000', '1379009', '1405197', '1408710', '1420850', '1574565']

    cik = '1420850'
    print("{}:{}".format(cik, firmname(cik)))
    tf = testfiles(cik)
    price_range = [get_price_range(f) for f in tf]
    print_pricerange(cik)
