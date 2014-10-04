# /usr/bin/python3
# encoding: utf-8

import csv, time, traceback, gzip, io
from lxml           import etree
from pprint         import pprint

from itertools      import count
from functools      import reduce
from operator       import iadd
from IPython        import embed
import pandas as pd
import arrow
import requests



# retuns all rows for a given month
def get_rows(tab, month):
    print("Getting section: <{}> for <{}>".format(tab, month))
    url = "http://www.nasdaq.com/markets/ipos/activity.aspx"
    url += "?tab={T}&month={M}".format(T=tab, M=month)
    print(url)
    sess = requests.get(url)
    xml = etree.HTML(sess.text)

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



def update_nasdaq_pricings(endmonth='2014-09'):

    df_pricings = pd.read_csv("data/nasdaq_pricings.csv")
    # Check last date from pricings, begin from there.
    month, d, year = list(df_pricings.tail()['Date Priced'])[0].split('/')
    startmonth = '0'+ month if len(month)==1 else month
    startmonth = year + '-' + startmonth

    months = month_range(startmonth, endmonth)
    for month in months:
        print("Start month: {} -> End month: {}".format(startmonth, endmonth))
        pricings.extend(get_rows('pricings', month))

    new_pricings = pd.DataFrame(pricings[1:], columns=pricings[0])
    df_pricings = pd.read_csv("data/nasdaq_pricings.csv")
    df_pricings = df_pricings.merge(new_pricings, how='outer')
    df_pricings = df_pricings[df_pricings['Market'] != 'OTCBB']
    df_pricings.to_csv("data/nasdaq_pricings.csv", index=False)






# collects and saves data into csv
if __name__ == "__main__":

    months = month_range('2005-01', '2014-08')

    pricings_file = 'pricings.csv'
    filings_file = 'filings.csv'
    withdrawns_file = 'withdrawns.csv'
    pricings = [['Company Name', 'Symbol', 'Market', 'Price', 'Shares', 'Offer Amount', 'Date Priced', 'URL']]
    filings = [['Company Name', 'Symbol', 'Offer Amount', 'Date Filed', 'URL']]
    withdrawns = [['Company Name', 'Symbol', 'Market', 'Price', 'Shares', 'Offer Amount', 'Date Filed', 'Date Withdrawn', 'URL']]

    # for month in months:
    #   pricings.extend(get_rows('pricings', month))
    #   filings.extend(get_rows('filings', month))
    #   withdrawns.extend(get_rows('withdrawn', month))

    # with open(pricings_file, 'wb') as csvfile:
    #   csv.writer(csvfile).writerows(pricings)

    # with open(filings_file, 'wb') as csvfile:
    #   csv.writer(csvfile).writerows(filings)

    # with open(withdrawns_file, 'wb') as csvfile:
    #   csv.writer(csvfile).writerows(withdrawns)


    # new_pricings = pd.DataFrame(pricings[1:], columns=pricings[0])
    # df_pricings = pd.read_csv("data/nasdaq_pricings.csv")
    # # df_filings = pd.read_csv("../SEC_index/filings.csv")
    # # df_withdrawn = pd.read_csv("../SEC_index/withdrawns.csv")

    # df_pricings = df_pricings.append(new_pricings)
    # df_pricings = df_pricings[df_pricings['Market'] != 'OTCBB']
    # df_pricings.to_csv("data/nasdaq_pricings.csv", index=False)







