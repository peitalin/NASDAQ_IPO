# /usr/bin/python3
# encoding: utf-8

import os
import csv, time, traceback, gzip, io
from urllib.request import Request, urlopen
from lxml           import etree
from pprint         import pprint

from itertools      import count
from functools      import reduce
from operator       import iadd
from IPython        import embed
import pandas as pd
import arrow




def read_url(url, success_delay=0.5, error_delay=5):
    "returns html page, adds a delay and retry if needed"

    tries = count(1)
    while next(tries) < 5:
        try:
            request = Request(url)
            request.add_header('Accept-encoding', 'gzip')
            response = urlopen(request)
            if response.info().get('Content-Encoding') == 'gzip':
                buf = io.BytesIO(response.read())
                f = gzip.GzipFile(fileobj=buf)
                data = f.read()
                print("Read GZip Data", url)
            else:
                data = response.read()
                print("Read Uncompressed Data")
            time.sleep(success_delay)
            return data
        except:
            traceback.print_exc()
            time.sleep(error_delay)



# retuns all rows for a given month
def get_rows(tab, month):
    print("Getting section: <{}> for <{}>".format(tab, month))
    url = "http://www.nasdaq.com/markets/ipos/activity.aspx"
    url += "?tab={T}&month={M}".format(T=tab, M=month)
    print(url)
    xml = etree.HTML(read_url(url))

    rows = []
    trs = xml.xpath('//div[@class="genTable"]/table/tbody/tr')
    for tr in trs:
        tds = tr.xpath('./td')
        row = reduce(iadd, (td.xpath(".//text()") for td in tds))
        url = tr.xpath('./td[1]/a/@href')[0]
        row.append(url)
        rows.append(row)
    return rows


def month_range(start, end):
    date_range = arrow.Arrow.range('month', arrow.get(start), arrow.get(end))
    return [a.strftime('%Y-%m') for a in date_range]


# collects and saves data into csv
if __name__ == "__main__":

    months = month_range('2005-01', '2015-10')

    pricings = [['Company Name', 'Symbol', 'Market', 'Price', 'Shares', 'Offer Amount', 'Date Priced', 'URL']]
    filings = [['Company Name', 'Symbol', 'Offer Amount', 'Date Filed', 'URL']]
    withdrawns = [['Company Name', 'Symbol', 'Shares', 'Offer Amount', 'Date Filed', 'Date Withdrawn', 'URL']]

    for month in months:
      pricings.extend(get_rows('pricings', month))
      filings.extend(get_rows('filings', month))
      withdrawns.extend(get_rows('withdrawn', month))

    withdrawns = [[x[0], ''] + x[1:] if len(x)==6 else x for x in withdrawns]
    # Some are missing ticker symbols

    pd.DataFrame(pricings[1:], columns=pricings[0]).to_csv('data/nasdaq_pricings.csv')
    pd.DataFrame(filings[1:], columns=filings[0]).to_csv('data/nasdaq_filings.csv')
    pd.DataFrame(withdrawns[1:] , columns=withdrawns[0]).to_csv('data/nasdaq_withdrawns.csv')











