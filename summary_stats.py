#!/usr/bin/env python


import os
import re
import sys
import json
import pandas as pd
import numpy as np
import arrow

from itertools import chain
from widgets import as_cash, next_row, next_col
from xlwings import Workbook, Range, Sheet

from scipy.stats.mstats import kruskalwallis
import scipy.stats as stats
import seaborn as sb
import matplotlib.pyplot as plt


FILE_PATH = 'text_files/'
IPO_DIR = os.path.join(os.path.expanduser("~"), "Data", "IPO")
BASEDIR = os.path.join(IPO_DIR, "NASDAQ")
FINALJSON = json.loads(open('final_json.txt').read())


conames_ciks = {cik:FINALJSON[cik]['Company Overview']['Company Name'] for cik in FINALJSON}
firmname = lambda cik: conames_ciks[cik]
get_cik = lambda firm: [x[0] for x in conames_ciks.items() if x[1].lower().startswith(firm)][0]
def aget(sdate):
    sdate = sdate if isinstance(sdate, str) else sdate.isoformat()
    if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', sdate):
        return arrow.get(sdate, 'M-D-YYYY') if '-' in sdate else arrow.get(sdate, 'M/D/YYYY')
    elif re.search(r'\d{4}[/-]\d{2}[/-]\d{2}', sdate):
        return arrow.get(sdate, 'YYYY-MM-DD') if '-' in sdate else arrow.get(sdate, 'YYYY/MM/DD')










def descriptive_stats():


    from xlwings import Workbook, Range, Sheet

    df['log_proceeds'] = np.log(df['proceeds'])
    keystats = [np.size, np.mean, np.std, np.min, np.median, np.max]
    kkeys = ['percent_first_price_update', 'number_of_price_updates', 'log_proceeds', 'market_cap', 'share_overhang', 'EPS', 'liab_over_assets', 'underwriter_rank_avg', 'M3_indust_rets', 'BAA_yield_changes', 'open_return', 'close_return', 'prange_change_first_price_update', 'underwriter_num_leads', ]

    def stratified(groupby, df):
        return [x[1] for x in df.groupby(groupby)]

    def kwtest(s, groupby, df):
        return kruskalwallis(*[group[s] for group in stratified(groupby, df)])

    def f_test(s, groupby, df):
        return stats.f_oneway(*[group[s] for group in stratified(groupby, df)])

    def l_test(s, groupby, df):
        return stats.levene(*[group[s] for group in stratified(groupby, df)])

    ### XLS API Functions

    def above_within_under(sample):

        wb = Workbook("xlwings.xls")
        kwdict = {
            'percent_first_price_update': 'Percent 1st Price Update',
            'number_of_price_updates': 'No. Price Updates',
            'prange_change_first_price_update': 'Price Range (Max - Min)',
            'market_cap': 'Market Cap (mil)',
            'log_proceeds': 'ln(Proceeds)',
            'share_overhang': 'Share Overhang',
            'EPS': 'EPS',
            'liab_over_assets': 'Liab/Assets',
            'underwriter_rank_avg': 'Underwriter Rank',
            'M3_indust_rets': 'Industry Returns',
            'open_return': 'Price Jump',
            'close_return': 'Initial Return',
            'prange_change_first_price_update': 'Price Range (Max - Min)'
            }

        # Correct Order
        kwkeys = ['percent_first_price_update', 'number_of_price_updates', 'prange_change_first_price_update', 'market_cap', 'log_proceeds', 'M3_indust_rets',  'liab_over_assets', 'EPS', 'share_overhang', 'underwriter_rank_avg', ]

        s1 = df[df['size_of_final_price_revision'].notnull()][
                    ['offer_in_filing_price_range', 'underwriter_tier'] + kkeys]
        s1['market_cap'] = s1['market_cap'] / 1000000 # Mils
        s1['M3_indust_rets'] /= 100
        s1['percent_first_price_update'] /= 100

        sumstat = s1.groupby(['offer_in_filing_price_range']) \
                        .agg(keystats) \
                        .reindex(index=['above','within','under'])
        sumstat.index = ['Above', 'Within', 'Under']

        kwstats = {key:kwtest(key, 'offer_in_filing_price_range', s1) for key in kwkeys}
        l_stats = {key:l_test(key, 'offer_in_filing_price_range', s1) for key in kwkeys}
        f_stats = {key:f_test(key, 'offer_in_filing_price_range', s1) for key in kwkeys}

        Sheet("above_within_under").activate()
        Range("A3:T32").value = [['']*44]*44
        Range("C4").value = ['obs', 'mean', 'std', 'min', 'med', 'max', 'ANOVA']
        Range("L4").value = ['', 'mean', 'std', 'min', 'med', 'max', 'ANOVA']

        nrows = 2 + len(sumstat.index)
        cells = chain(*[('B%s' % s, 'K%s' % s) for s in range(5, len(kwkeys)*nrows, nrows)])
        # ['B5', 'B10', 'B15', 'B20', 'K5', 'K10', 'K15', 'K20']

        for i, cell, key in zip(range(len(kwkeys)), cells, kwkeys):
            Range(cell).value = sumstat[key]
            Range(cell).value = ['', kwdict[key] ,'','','','','','','']
            kw_cell = next_row(next_col(cell, 7), len(sumstat.index))

            if l_stats[key][0] != l_stats[key][0]:
                # Test nan
                use_kw_test = True
            elif l_stats[key][1] > 0.05:
                use_kw_test = True
            else:
                use_kw_test = False

            if use_kw_test:
                # Use Kruskall Wallace Test
                Range(kw_cell).value = 'H-stat'
                Range(next_row(kw_cell)).value = kwstats[key][0]
                Range(next_col(kw_cell)).value = 'p-value'
                Range(next_row(next_col(kw_cell))).value = kwstats[key][1]
            else:
                # Use F-test
                Range(kw_cell).value = 'F-stat'
                Range(next_row(kw_cell)).value = f_stats[key][0]
                Range(next_col(kw_cell)).value = 'p-value'
                Range(next_row(next_col(kw_cell))).value = f_stats[key][1]

            # Clean up columns
            if cell.startswith('K'):
                Range('%s:%s' % (cell, next_row(cell, 4))).value = [['']]*4
            if i != 0:
                Range(next_col(next_row(cell))).value = [['']]*3


    def full_vs_filtered_sample(sample):

        wb = Workbook("xlwings.xls")
        fulldf = pd.read_csv('full_df.csv', dtype={'cik': object})
        fulldf.set_index('cik', inplace=True)
        fulldf['market_cap'] = fulldf['market_cap'] / 1000000 # Mils
        fulldf['filtered'] = [cik in df.index for cik in fulldf.index]

        fulldf['M3_indust_rets'] = fulldf['2month_indust_rets']

        kwdict = {
            'percent_first_price_update': 'Percent 1st Price Update',
            'number_of_price_updates': 'No. Price Updates',
            'market_cap': 'Market Cap (mil)',
            'log_proceeds': 'ln(Proceeds)',
            'share_overhang': 'Share Overhang',
            'EPS': 'EPS',
            'liab/assets': 'Liab/Assets',
            'underwriter_rank_avg': 'Underwriter Rank',
            'underwriter_num_leads': 'No. Lead Underwriters',
            'M3_indust_rets': 'Industry Returns',
            'open_return': 'Price Jump',
            'close_return': 'Initial Return',
            'prange_change_first_price_update': 'Price Range (Max - Min)'
            }

        # Correct Order
        kwkeys = ['percent_first_price_update', 'number_of_price_updates', 'prange_change_first_price_update', 'market_cap', 'log_proceeds', 'M3_indust_rets', 'liab/assets', 'EPS', 'share_overhang', 'underwriter_rank_avg', 'open_return', 'close_return', ]


        kwstats = {key:kwtest(key, 'filtered', df=fulldf[fulldf[key].notnull()])
                        for key in kwkeys}
        l_stats = {key:l_test(key, 'filtered', df=fulldf[fulldf[key].notnull()])
                        for key in kwkeys}
        f_stats = {key:f_test(key, 'filtered', df=fulldf[fulldf[key].notnull()])
                        for key in kwkeys}

        Sheet("sample_bias").activate()
        Range("A3:T32").value = [['']*44]*44
        Range("C4").value = ['obs', 'mean', 'std', 'min', 'med', 'max', 'ANOVA']
        Range("L4").value = ['', 'mean', 'std', 'min', 'med', 'max', 'ANOVA']

        nrows = 2 + len(set(fulldf['filtered']))
        cells = list(chain(*[('B%s' % s, 'K%s' % s) for s in range(5, len(kwkeys)*nrows, nrows)]))
        # ['B5', 'B10', 'B15', 'B20', 'K5', 'K10', 'K15', 'K20']

        for i, cell, key in zip(range(len(kwkeys)), cells, kwkeys):

            sumstat = fulldf[fulldf[key].notnull()].groupby(['filtered']) \
                                                    .agg(keystats) \
                                                    .reindex(index=[True,False])
            sumstat.index = ['Final Sample', 'Filtered']

            Range(cell).value = sumstat[key]
            Range(cell).value = ['', kwdict[key] ,'','','','','','','']
            kw_cell = next_row(next_col(cell, 7), len(sumstat.index))

            if l_stats[key][0] != l_stats[key][0]:
                # Test nan
                use_kw_test = True
            elif l_stats[key][1] > 0.05:
                use_kw_test = True
            else:
                use_kw_test = False

            if use_kw_test:
                # Use Kruskall Wallace Test
                Range(kw_cell).value = 'H-stat'
                Range(next_row(kw_cell)).value = kwstats[key][0]
                Range(next_col(kw_cell)).value = 'p-value'
                Range(next_row(next_col(kw_cell))).value = kwstats[key][1]
            else:
                # Use F-test
                Range(kw_cell).value = 'F-stat'
                Range(next_row(kw_cell)).value = f_stats[key][0]
                Range(next_col(kw_cell)).value = 'p-value'
                Range(next_row(next_col(kw_cell))).value = f_stats[key][1]

            # Clean up columns
            if cell.startswith('K'):
                Range('%s:%s' % (cell, next_row(cell, 4))).value = [['']]*4
            # if i != 0:
            #     Range(next_col(next_row(cell))).value = [['']]*3


    def attention_strata(sample):

        wb = Workbook("xlwings.xls")
        # Correct Order
        kwkeys = [
            'IoT_15day_CASI_all',
            'IoT_30day_CASI_all',
            'IoT_15day_CASI_weighted_finance',
            'IoT_30day_CASI_weighted_finance',
            'IoT_15day_CASI_news',
            'IoT_30day_CASI_news',
            ]

        sumstat_index = {
            'amends': {
                'sheet': 'attention_amend',
                'order': ['Up', 'None', 'Down'],
                'rename': ['Up', 'None', 'Down']
            },
            'offer_in_filing_price_range': {
                'sheet': 'attention_revise',
                'order': ['above','within','under'],
                'rename': ['Above', 'Within', 'Under']
            },
            'underwriter_tier': {
                'sheet': 'attention_uw',
                'order': ['8.5+', '7+', '0+'],
                'rename': ['8.5+', '7+', '0+']
            },
            'VC': {
                'sheet': 'attention_vc',
                'order': [1, 0],
                'rename': ['VC', 'No VC']
            },
            'exchange': {
                'sheet': 'attention_exchange',
                'order': ['New York Stock Exchange','NASDAQ','American Stock Exchange'],
                'rename': ['NYSE', 'NASDAQ', 'AMEX']
            }
        }

        kwdict = {
            'IoT_15day_CASI_all': "15-Day CASI (All)",
            'IoT_30day_CASI_all': "30-Day CASI (All)",
            'IoT_15day_CASI_weighted_finance': "15-Day CASI (Weighted Finance)",
            'IoT_30day_CASI_weighted_finance': "30-Day CASI (Weighted Finance)",
            'IoT_15day_CASI_news': "15-Day CASI (Business News)",
            'IoT_30day_CASI_news': "30-Day CASI (Business News)",
            }


        gkey = 'offer_in_filing_price_range'
        gkey = 'underwriter_tier'
        # gkey = 'amends'
        # gkey = 'VC'
        # gkey = 'exchange'

        if gkey=='amends':
            df = pd.read_csv("df_update.csv", dtype={'cik':object})
        else:
            df = pd.read_csv("df.csv", dtype={'cik':object})

        attention = df[[gkey] + kwkeys]
        if gkey=='underwriter_tier':
            attention = attention[attention['underwriter_tier'] != '-1']
        if gkey=='exchange':
            attention['exchange'] = ['NASDAQ' if 'NASDAQ' in x else x for x in attention['exchange']]

        kwstats = {key:kwtest(key, gkey, df=attention) for key in kwkeys}
        l_stats = {key:l_test(key, gkey, df=attention) for key in kwkeys}
        f_stats = {key:f_test(key, gkey, df=attention) for key in kwkeys}

        Sheet(sumstat_index[gkey]['sheet']).activate()
        Range("A2:T66").value = [['']*66]*66
        Range("C4").value = ['obs', 'mean', 'std', 'min', 'med', 'max', 'ANOVA']
        Range("L4").value = ['', 'mean', 'std', 'min', 'med', 'max', 'ANOVA']

        num_category = len(set(attention[gkey]))
        nrows = 2 + num_category
        cells = list(chain(*[('B%s' % s, 'K%s' % s) for s in range(5, len(kwkeys)*nrows, nrows)]))
        # ['B5', 'B10', 'B15', 'B20', 'K5', 'K10', 'K15', 'K20']

        for i, cell, key in zip(range(len(kwkeys)), cells, kwkeys):

            sumstat = attention.groupby([gkey]) \
                                .agg(keystats) \
                                .reindex(index=sumstat_index[gkey]['order'])
            sumstat.index = sumstat_index[gkey]['rename']

            Range(cell).value = sumstat[key]
            Range(cell).value = ['', kwdict[key] ,'','','','','','','']
            kw_cell = next_row(next_col(cell, 7), len(sumstat.index))

            if l_stats[key][0] != l_stats[key][0]:
                # Test nan
                use_kw_test = True
            elif l_stats[key][1] > 0.05:
                use_kw_test = True
            else:
                use_kw_test = False

            if use_kw_test:
                # Use Kruskall Wallace Test
                Range(kw_cell).value = 'H-stat'
                Range(next_row(kw_cell)).value = kwstats[key][0]
                Range(next_col(kw_cell)).value = 'p-value'
                Range(next_row(next_col(kw_cell))).value = kwstats[key][1]
            else:
                # Use F-test
                Range(kw_cell).value = 'F-stat'
                Range(next_row(kw_cell)).value = f_stats[key][0]
                Range(next_col(kw_cell)).value = 'p-value'
                Range(next_row(next_col(kw_cell))).value = f_stats[key][1]

            # Clean up columns
            if cell.startswith('K'):
                Range('%s:%s' % (cell, next_row(cell, 4))).value = [['']]*4
            if i != 0:
                Range(next_col(next_row(cell))).value = [['']] * num_category






if __name__=='__main__':

    df = pd.read_csv('df.csv', dtype={'cik':object})
    df.set_index('cik', inplace=True)
    cik = '1326801' # Facebook


    # df['percent_first_price_update'] = [x if x==x else 0 for x in df['percent_first_price_update']]
    amendments = df[~df.size_of_first_price_update.isnull()]
    revisions = df[~df.size_of_final_price_revision.isnull()]

    # # check .describe() to see key order: above, under, within (alphabetical)
    # above, under, within = [x[1] for x in amendments.groupby('offer_in_filing_price_range')]
    # noupdate, update = [x[1] for x in revisions.groupby('amends')]






