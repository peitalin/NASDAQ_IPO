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
firmname = lambda cik: FINALJSON[cik]['Company Overview']['Company Name']

def extract_NASDAQ_IPO_data_threaded(links):
    """Multi-threaded wrapper for NSADAQ Data scraping.
    Scrapes all data from NASDAQ IPO links obtained from get_nasdaq_links.py.
    """

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

    with ThreadPoolExecutor(max_workers=N) as exec:
        json_result = exec.map(extract_NASDAQ_IPO_data, [iter(links)]*N)
        final_dict  = reduce(lambda d1, d2: dict(d1, **d2), json_result)
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




def get_s1_filings(FINALJSON):
    """Downloads all filings for all firms (ciks) in FINALJSON.
    Makes system calls to wget, downloads gzipped filings and extracts.
    On keyboard interupt, deletes half downloaded file
    """
    import subprocess
    import gzip

    filings = {cik:FINALJSON[cik]['Filing'] for cik,vals in FINALJSON.items()}
    error_urls = []
    filings_ = list(filings.items())
    # cik = 1326801; filelist = FINALJSON[cik]['Filing'][0] # FB test
    for cik, filelist in filings_:
        for filing in filelist:
            try:
                url = 'http://www.nasdaq.com/markets/ipos/' + filing[3].replace('/markets/ipos/','')
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
                usr_agent = 'python-requests/2.2.1 CPython/3.4.0 Darwin/13.3.0'
                command = '''wget \
                            --header="User-Agent: {usr_agent}" \
                            --header="Connection: Keep-Alive"  \
                            --header="Accept-Encoding: gzip, deflate, compress" \
                            --header="Accept: */*" \
                            "{url}"  '''.format(usr_agent=usr_agent, url=url)

                exit_status = os.system(command)
                if exit_status==0:
                    content = gzip.open(filename, 'rb').read()
                    with open(filename, 'wb') as f:
                        f.write(content)
                        print('=> Unzipped {} as HTML file of size: {:,} kbs\n\n'.format(
                            filename[-28:],os.path.getsize(filename)))
                    continue

                elif exit_status==2048:
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
    "Conforms headings on JSON data downloaded from NASDAQ"

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

    for cik in FINALJSON:
        assert len(cik) == 7
        FINALJSON[cik]['Metadata']['CIK'] = cik

    return FINALJSON



if __name__=='__main__':
    df_pricings = pd.read_csv("./data/nasdaq_pricings.csv")
    finaljson_firms = [FINALJSON[cik]['Company Overview']['Company Name'] for cik in FINALJSON.keys()]
    df =  df_pricings[[x not in finaljson_firms for x in df_pricings['Company Name']]]
    # df = df_pricings[df_pricings['CIK'] == 'NA']
    # df.set_index('CIK', inplace=True)
    links = list(zip(df.index, df['URL']))
    # final_dict = extract_NASDAQ_IPO_data_threaded(links)


    FULLJSON = json.loads(open('full_json.txt').read())
    # with open('full_json.txt', 'w') as f:
    #     f.write(json.dumps(FULLJSON, indent=4, sort_keys=True))


    # ### Writing json file
    # with open('final_dict.txt', 'w') as myfile:
    #   myfile.write(json.dumps(final_dict, indent=4, sort_keys=True))

    # with open('final_json.txt', 'w') as f:
    #     f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))






    ######## NEW BAD INDUSTRIES ######################################
    # bad_sic = [
    #     '6021', '6022', '6035', '6036', '6111',   '6153',
    #     '6159', '6162', '6163', '6172', '6189', '6200', '6022',
    #     '6221', '6770', '6792', '6794', '6795', '6798', '6799',
    #     '8880', '8888', '9721', '9995'
    #     ]
    # {cik:firmname(cik) for cik in FINALJSON if FINALJSON[cik]['Metadata']['SIC'] == '6021'}
    # FINALJSON = {cik:vals for cik,vals in FINALJSON.items()
    #              if vals['Metadata']['SIC'] not in bad_sic}
    ###################################################################


    ########## BAD IPOS
    ###### No share offerings?
    # bad_ipos: ['1022345', '1314475', '1411303']
    # hidden unit trusts
    # ['1255474', '1557421'] # WHITING PETROLEUM CORP, Igynta



    ######## SIC CODES
    # SIC_Code|AD_Office|Industry_Title
    # 6021|7|NATIONAL COMMERCIAL BANKS
    # 6022|7|STATE COMMERCIAL BANKS
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


