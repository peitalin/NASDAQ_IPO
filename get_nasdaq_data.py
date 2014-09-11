# /usr/bin/python3


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
USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/21.0'
FINALJSON = json.loads(open('final_json.txt').read())
aget = lambda datestr: arrow.get(datestr, 'M/D/YYYY')


def extract_NASDAQ_IPO_data_threaded(links):
    """Multi-threaded wrapper for NSADAQ Data scraping.
    Scrapes all data from NASDAQ IPO links obtained from get_nasdaq_links.py.
    """

    with ThreadPoolExecutor(max_workers=N) as exec:
        json_result = exec.map(extract_NASDAQ_IPO_data, [iter(links)]*N)
        final_dict  = reduce(lambda d1, d2: dict(d1, **d2), json_result)
    return final_dict

    def extract_NASDAQ_IPO_data(links):
        """Main loop called from extract_NASDAQ_IPO_data_threaded()"""

        final_dict = {}
        for link in links:
            if glob.glob(FILE_PATH + '/{}*'.format(str(link[0]))):
                print("NASDAQ data exists for {}, skipping".format(str(link[0])))
                continue

            print("Getting data for: " + link[1].split('/')[-1])
            url = overview_url = link[1]
            financials_url = url + "?tab=financials"
            experts_url = url + "?tab=experts"
            item_dict = {}

            company_overview_dict = scrape_company_overview(url)
            item_dict['Company Overview'] = company_overview_dict
            cik = company_overview_dict['CIK'][3:]

            item_dict['Metadata'] = {}
            item_dict['Metadata']['Use of Proceeds'] = cik + '_use_of_proceeds.txt'
            item_dict['Metadata']['Company Description'] = cik + '_company_description.txt'

            financials, filings = scrape_financials_and_filings(financials_url)
            item_dict['Financials'] = financials
            item_dict['Filing'] = filings
            item_dict['Experts'] = scrape_experts(experts_url)

            final_dict[cik] = item_dict
            time.sleep(random.randint(1,3))
        return final_dict


def scrape_company_overview(overview_url):

    ### Reading Company Overview tab
    try:
        html = requests.get(overview_url).text
        xml = etree.HTML(html)

        company_overview_dict = {}
        company_overview_rows = xml.xpath('//div[@id="infoTable"]/table//tr')
        for row in company_overview_rows:
            company_overview_dict[row.xpath('.//td[1]/text()')[0]] = ' '.join(row.xpath('.//td[2]//text()'))
        cik = company_overview_dict['CIK'][3:]

        ### Reading use of proceeds
        use_of_proceeds = "".join(xml.xpath('//div[@class="infoTable_2"]/pre/text()'))
        with open(FILE_PATH +  cik + '_use_of_proceeds.txt', 'wb') as f:
            f.write(use_of_proceeds.encode('utf-8'))

        ### Reading company description
        describe_main = xml.xpath('//div[@class="ipo-comp-description"]/pre/text()')[0]
        describe_more = xml.xpath('//div[@id="read_more_div_toggle1"]/pre/text()')[0]
        company_description = describe_main + describe_more
        with open(FILE_PATH + cik + '_company_description.txt', 'wb') as f:
            f.write(company_description.encode('utf-8'))

        return company_overview_dict

    except:
        print('Error while reading overview tab for:' + overview_url)
        traceback.print_exc()

def scrape_financials_and_filings(financials_url):

    try:
        html = requests.get(financials_url).text
        xml = etree.HTML(html)

        financials_dict = {}
        financials_rows = []
        fxpath = '//div[@class="genTable"]/div[@class="{X}"]/table//tr'
        for xtable in ["floatL width marginR15px", "floatL width"]:
            financials_rows += xml.xpath(fxpath.format(X=xtable))

        if not financials_rows:
            for xtable in ["left-table-wrap", "right-table-wrap"]:
                financials_rows += xml.xpath(fxpath.format(X=xtable))

        for row in financials_rows:
            rkey = row.xpath('.//td[1]//text()')[0]
            financials_dict[rkey] = row.xpath('.//td[2]//text()')[0]

        ### Reading filing
        filing_rows = xml.xpath('//div[@class="tab2"]/div[@class="genTable"]/table//tr')
        filings = []
        for row in filing_rows:
            if not row.xpath('.//td/text()')[:3]:
                continue
            filings.append(row.xpath('.//td/text()')[:3] + ["".join(row.xpath('.//td[4]/a/@href'))])
    except:
        print('Error while reading financials tab for:' + cik)
        traceback.print_exc()

    return financials_dict, filings

def scrape_experts(experts_url):
    ### Reading experts tab
    html = requests.get(experts_url).text
    xml = etree.HTML(html)
    experts_dict = {}
    experts_rows = xml.xpath('//div[@class="tab3"]/div[@class="genTable"]/table//tr')

    for row in experts_rows:
        # row_url = row.xpath('.//td/a')[0].attrib['href'] if row.xpath('.//td/a') else ''
        exp_type = row.xpath('.//td[1]/text()')[0]
        if exp_type in experts_dict:
            experts_dict[exp_type] = experts_dict[exp_type] + [row.xpath('.//td[2]//text()')[0]]
        else:
            experts_dict[exp_type] = [row.xpath('.//td[2]//text()')[0]]

    return experts_dict

def extract_SIC(final_dict):
    "Extracts SIC codes from SEC edgar"

    with ThreadPoolExecutor(max_workers=N) as exec:
        json_result = exec.map(extract_NASDAQ_IPO_data, [iter(links)]*N)
        final_dict  = reduce(lambda d1, d2: dict(d1, **d2), json_result)
    return final_dict

    def scrape_SIC(cik):
        url = 'https://www.sec.gov/cgi-bin/browse-edgar?CIK={}&Find=Search&owner=exclude&action=getcompany'.format(cik)
        res = requests.get(url)
        SIC = etree.HTML(res.text).xpath('//html/body/div/div/div/p/a/text()')[0]
        if 3 < len(SIC) < 8:
            return SIC

    ciks = final_dict.keys()
    no_sic = []
    for cik in ciks:
        if 'Metadata' in final_dict[cik].keys():
            if 'SIC code' in final_dict[cik]['Metadata']:
                if final_dict[cik]['Metadata']['SIC code'] != "NA":
                    print('Already had SIC', final_dict[cik]['Metadata']['SIC code'])
                    continue

        SIC_code = scrape_SIC(cik)
        if SIC_code:
            if 'Metadata' not in final_dict[cik]:
                final_dict[cik]['Metadata'] = {'SIC code': 'NA'}
            final_dict[cik]['Metadata']['SIC code'] = SIC_code
            print("SEC Edgar ->", final_dict[cik]['Metadata']['SIC code'])
            continue
    return final_dict

def get_s1_filings(FINALJSON):

    import subprocess
    filings = {cik:FINALJSON[cik]['Filing'] for cik,vals in FINALJSON.items()}

    error_urls = []
    filings_ = list(filings.items())
    for cik, filelist in filings_:
        for filing in filelist:
            try:
                url = 'www.nasdaq.com/markets/ipos/' + filing[3]
                savedir = os.path.join(BASEDIR , "filings", cik)
                filename = os.path.join(savedir, filing[3].split('/')[-1])
            except AttributeError:
                raise(AttributeError("3rd item in each filing must be URL"))

            if not os.path.exists(savedir):
                os.makedirs(savedir)

            if glob.glob(filename):
                filesize = os.path.getsize(filename)
                if filesize < 5120:
                    print("Filesize: {}".format(os.path.getsize(filename)))
                    os.remove(filename) # rm file < 5kb
                else:
                    print('>> Already downloaded {}: {}kb'.format(url, filesize))
                    continue

            # NASDAQ filings are tagged and cleaner than EDGAR filings.
            try:
                os.chdir(savedir)
                usr_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/21.0'
                command = 'wget --user-agent="{}" "{}"'.format(usr_agent, url)
                exit_status = os.system(command)
                if exit_status == 0:
                    continue
                elif exit_status == 2048:
                    error_urls.append((cik, url))
                else:
                    if os.path.exists(filename):
                        print("\nDeleting unfinished file: {}".format(filename))
                        os.remove(filename)
                    print("exit status: {}".format(exit_status))
                    raise(Exception("Keyboard Interrupt"))

            except KeyboardInterrupt as e:
                if os.path.exists(filename):
                    os.remove(filename)
                raise(e)
            time.sleep(round(random.uniform(0.2, 0.5), 2))

    os.chdir(BASEDIR)
    print('Error 500 internal service error URLS:')
    print(error_urls)



def final_json_clean_data(FINALJSON=FINALJSON):

    for cik in FINALJSON:
        if 'Metadata' not in FINALJSON[cik]:
            FINALJSON[cik]['Metadata'] = {}

    # Fixes Text File descriptions
    for cik in FINALJSON:
        FINALJSON[cik]['Metadata']['Company Description'] = cik + '_company_description.txt'
        FINALJSON[cik]['Metadata']['Use of Proceeds'] = cik + '_use_of_proceeds.txt'
        if 'Company_description' in FINALJSON[cik].keys():
            FINALJSON[cik].pop('Company_description')
        if 'Use of Proceeds' in FINALJSON[cik].keys():
            FINALJSON[cik].pop('Use of Proceeds')


    # Fixes 'Employees (as of MM-DD-YYYY)'
    for cik in FINALJSON:
        empkey = [x for x in FINALJSON[cik]['Company Overview'].keys() if x.startswith('Employees')][0]
        FINALJSON[cik]['Company Overview']['Employees'] = FINALJSON[cik]['Company Overview'][empkey]
        if empkey != 'Employees':
            FINALJSON[cik]['Company Overview'].pop(empkey)

    for cik in FINALJSON:
        FINALJSON[cik]['Metadata']['Number of Filings'] = len(FINALJSON[cik]['Filing'])

    return FINALJSON




    # for cik in FINALJSON:
    #     FINALJSON[cik]['Metadata']['Days from Pricing to Listing'] = len(FINALJSON[cik]['Filing'])
    #     first_pricing = aget(FINALJSON[cik]['Filing'][0][2])
    #     second_pricing = aget(FINALJSON[cik]['Filing'][XXX][2])

    # with open('final_json.txt', 'w') as f:
    #     f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))





if __name__=='__main__':
    df_pricings = pd.read_csv("./data/nasdaq_pricings.csv")
    finaljson_firms = [FINALJSON[cik]['Company Overview']['Company Name'] for cik in FINALJSON.keys()]
    df =  df_pricings[[x not in finaljson_firms for x in df_pricings['Company Name']]]
    # df = df_pricings[df_pricings['CIK'] == 'NA']
    # df.set_index('CIK', inplace=True)
    links = list(zip(df.index, df['URL']))
    # final_dict = extract_NASDAQ_IPO_data_threaded(links)

    # ### Writing json file
    # with open('final_dict.txt', 'w') as myfile:
    #   myfile.write(json.dumps(final_dict, indent=4, sort_keys=True))

    # with open('final_json.txt', 'w') as f:
    #     f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))



    # X1) Get S-1 filings from NASDAQ
    # X2) GET SIC codes from edgar / rawjson.txt
    # X3) Filter bad SIC codes
    # X4) Match CRSP/Yahoo Firms with NASDAQ firms for opening/close prices
    # X5) Get WRDS access and US CRSP dataset for stock prices.
    # X6) Fix Gtrends names, start scraping attention
    # X7) Make sure equity offer is actually IPO (check CUSIP)
    # X8) Get partial price adjustments and put in "Filings"
    # 9) Check Spinoff IPOs




    # share overhang
    # SIC industry returns
    # IPO cycle variables
    # Gtrends variables
    # underwriter rank (average lead underwriters)
    # no. underwriters
    # VC dummy (crunchbase)
    # confidential IPO
    # EPS





############ Remove BAD EXPERTS
    # badciks = []
    # badnames = []
    # for cik in FINALJSON.keys():
    #     try:
    #         islink = FINALJSON[cik]['Experts']['Auditor'][1]
    #         if "http://www.nasdaq.com/markets/ipos" in islink:
    #             badciks.append(cik)
    #             badnames.append(FINALJSON[cik]['Company Overview']['Company Name'])
    #             print(FINALJSON[cik]['Company Overview']['Company Name'],': wrong experts')
    #     except IndexError:
    #         continue


    # for cik, firm in zip(badciks, badnames):
    #     i, url = next(df[df['Company Name']==firm]['URL'].items())
    #     experts_url = url + "?tab=experts"
    #     experts_dict = scrape_experts(experts_url)
    #     print(experts_dict)
    #     FINALJSON[cik]['Experts'] = experts_dict





######## SIC CODES
# SIC_Code|AD_Office|Industry_Title
# 6035|7|SAVINGS INSTITUTION, FEDERALLY CHARTERED
# 6036|7|SAVINGS INSTITUTIONS, NOT FEDERALLY CHARTERED
# 6099|7|FUNCTIONS RELATED TO DEPOSITORY BANKING, NEC
# 6111|12|FEDERAL & FEDERALLY-SPONSORED CREDIT AGENCIES
# 6153|7|SHORT-TERM BUSINESS CREDIT INSTITUTIONS
# 6159|7|MISCELLANEOUS BUSINESS CREDIT INSTITUTION
# 6162|7|MORTGAGE BANKERS & LOAN CORRESPONDENTS
# 6163|7|LOAN BROKERS
# 6172|7|FINANCE LESSORS
# 6189|OSF|ASSET-BACKED SECURITIES
# 6200|8|SECURITY & COMMODITY BROKERS, DEALERS, EXCHANGES & SERVICES
# 6221|8|COMMODITY CONTRACTS BROKERS & DEALERS
# 6770|All|BLANK CHECKS
# 6792|4|OIL ROYALTY TRADERS
# 6794|3|PATENT OWNERS & LESSORS
# 6795|9|MINERAL ROYALTY TRADERS
# 6798|8|REAL ESTATE INVESTMENT TRUSTS
# 6799|8|INVESTORS, NEC
