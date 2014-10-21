#!/usr/bin/env python


import os
import re
import sys
import json
import pandas as pd
import numpy as np
import arrow

from itertools import chain
from widgets import as_cash
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








if 'excel_cell_movement_functions':

    def next_row(char, n=1):
        "Shifts cell reference by n rows."

        is_xls_cell = re.compile(r'^[A-Z].*[0-9]$')
        if not is_xls_cell.search(char):
            raise(Exception("'{}' is not a valid cell".format(char)))

        if n == 0:
            return char

        idx = [i for i,x in enumerate(char) if x.isdigit()][0]
        if int(char[idx:]) + n < 0:
            return char[:idx] + '1'
        else:
            return char[:idx] + str(int(char[idx:]) + n)

    def next_col(char, n=1):
        "Shifts cell reference by n columns."

        is_xls_cell = re.compile(r'^[A-Z].*[0-9]$')
        if not is_xls_cell.search(char):
            raise(Exception("'{}' is not a valid cell".format(char)))

        if n == 0:
            return char

        def next_char(char):
            "Next column in excel"
            if all(c=='Z' for c in char):
                return 'A' * (len(char) + 1)
            elif len(char) == 1:
                return chr(ord(char) + 1)
            elif char.endswith('Z'):
                return next_char(char[:-1]) + 'A'
            else:
                return 'A' + next_char(char[1:])

        def prev_char(char):
            "Previous column in excel"
            if len(char) == 1:
                return chr(ord(char) - 1) if char != 'A' else ''
            elif not char.endswith('A'):
                return char[:-1] + prev_char(char[-1])
            elif char.endswith('A'):
                return prev_char(char[:-1]) + 'Z'

        idx = [i for i,x in enumerate(char) if x.isdigit()][0]
        row = char[idx:]
        col = char[:idx]
        for i in range(abs(n)):
            col = next_char(col) if n > 0 else prev_char(col)
        return col + row






def descriptive_stats():


    from xlwings import Workbook, Range, Sheet
    wb = Workbook("xlwings.xls")

    keystats = [np.size, np.mean, np.std, np.min, np.median, np.max]
    kkeys = ['percent_first_price_update', 'number_of_price_updates', 'log_proceeds', 'market_cap', 'share_overhang', 'EPS', 'liab/assets', 'underwriter_rank_avg', '3month_indust_rets', 'BAA_yield_changes', 'open_return', 'close_return', 'prange_change_first_price_update', 'underwriter_num_leads', ]

    sample = df[df['size_of_final_price_revision'].notnull()][
                ['offer_in_filing_price_range', 'underwriter_tier'] + kkeys]
    sample['market_cap'] = sample['market_cap'] / 1000000 # Mils

    def stratified(groupby, df):
        return [x[1] for x in df.groupby(groupby)]

    def kwtest(s, groupby, df):
        return kruskalwallis(*[group[s] for group in stratified(groupby, df)])

    def f_test(s, groupby, df):
        return stats.f_oneway(*[group[s] for group in stratified(groupby, df)])

    def l_test(s, groupby, df):
        return stats.levene(*[group[s] for group in stratified(groupby, df)])

    ### XLS API Functions

    def update_noupdate(sample):

        kwdict = {
            'prange_change_first_price_update': 'Price Range Change',
            'market_cap': 'Market Cap (mil)',
            'log_proceeds': 'ln(Proceeds)',
            'share_overhang': 'Share Overhang',
            'EPS': 'EPS',
            'liab/assets': 'Liab/Assets',
            'underwriter_rank_avg': 'Underwriter Rank',
            'underwriter_num_leads': 'No. Lead Underwriters',
            '2month_indust_rets': 'Industry Returns',
            'BAA_yield_changes': 'BAA Yield Change',
            'open_return': 'Price Jump',
            'close_return': 'Initial Return',
            }

        kwkeys = ['prange_change_first_price_update', 'market_cap', 'log_proceeds', '2month_indust_rets', 'BAA_yield_changes', 'liab/assets', 'EPS', 'share_overhang',  'underwriter_num_leads', 'underwriter_rank_avg', 'open_return', 'close_return', ]

        sumstat = sample.groupby('amends') \
                        .agg(keystats) \
                        .reindex(index=[True, False])
        sumstat.index = ['Update', 'No Update']

        kwstats = {key:kwtest(key, 'amends') for key in kwkeys}
        l_stats = {key:l_test(key, 'amends') for key in kwkeys}
        f_stats = {key:f_test(key, 'amends') for key in kwkeys}

        Sheet("update_noupdate").activate()
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


    def above_within_under(sample):

        kwdict = {
            'percent_first_price_update': 'Percent 1st Price Update',
            'number_of_price_updates': 'No. Price Updates',
            'prange_change_first_price_update': 'Price Range (Max - Min)',
            'market_cap': 'Market Cap (mil)',
            'log_proceeds': 'ln(Proceeds)',
            'share_overhang': 'Share Overhang',
            'EPS': 'EPS',
            'liab/assets': 'Liab/Assets',
            'underwriter_rank_avg': 'Underwriter Rank',
            'underwriter_num_leads': 'No. Lead Underwriters',
            '2month_indust_rets': 'Industry Returns',
            'BAA_yield_changes': 'BAA Yield Change',
            'open_return': 'Price Jump',
            'close_return': 'Initial Return',
            'prange_change_first_price_update': 'Price Range (Max - Min)'
            }

        # Corect Order
        kwkeys = ['percent_first_price_update', 'number_of_price_updates', 'prange_change_first_price_update', 'market_cap', 'log_proceeds', '2month_indust_rets', 'BAA_yield_changes', 'liab/assets', 'EPS', 'share_overhang',  'underwriter_num_leads', 'underwriter_rank_avg', 'open_return', 'close_return', ]

        sumstat = sample.groupby(['offer_in_filing_price_range']) \
                        .agg(keystats) \
                        .reindex(index=['above','within','under'])
        sumstat.index = ['Above', 'Within', 'Under']

        kwstats = {key:kwtest(key, 'offer_in_filing_price_range') for key in kwkeys}
        l_stats = {key:l_test(key, 'offer_in_filing_price_range') for key in kwkeys}
        f_stats = {key:f_test(key, 'offer_in_filing_price_range') for key in kwkeys}

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

        fulldf = pd.read_csv('full_df.csv', dtype={'cik': object})
        fulldf.set_index('cik', inplace=True)
        fulldf['market_cap'] = fulldf['market_cap'] / 1000000 # Mils
        fulldf['filtered'] = [cik in sample.index for cik in fulldf.index]

        kwdict = {
            'percent_first_price_update': 'Percent 1st Price Update',
            'number_of_price_updates': 'No. Price Updates',
            'prange_change_first_price_update': 'Price Range (Max - Min)',
            'market_cap': 'Market Cap (mil)',
            'log_proceeds': 'ln(Proceeds)',
            'share_overhang': 'Share Overhang',
            'EPS': 'EPS',
            'liab/assets': 'Liab/Assets',
            'underwriter_rank_avg': 'Underwriter Rank',
            'underwriter_num_leads': 'No. Lead Underwriters',
            '2month_indust_rets': 'Industry Returns',
            'BAA_yield_changes': 'BAA Yield Change',
            'open_return': 'Price Jump',
            'close_return': 'Initial Return',
            'prange_change_first_price_update': 'Price Range (Max - Min)'
            }

        # Correct Order
        kwkeys = ['percent_first_price_update', 'number_of_price_updates', 'prange_change_first_price_update', 'market_cap', 'log_proceeds', '2month_indust_rets', 'BAA_yield_changes', 'liab/assets', 'EPS', 'share_overhang',  'underwriter_num_leads', 'underwriter_rank_avg', 'open_return', 'close_return', ]


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

        # Correct Order
        kwkeys = [
            'IoT_15day_CASI_all', 'ln_15day_CASI_all',
            'IoT_30day_CASI_all', 'ln_30day_CASI_all',
            'IoT_60day_CASI_all', 'ln_60day_CASI_all'
            ]

        sumstat_index = {
            'amends': {
                'sheet': 'attention1',
                'order': ['Up', 'None', 'Down'],
                'rename': ['Up', 'None', 'Down']
                },
            'offer_in_filing_price_range': {
                'sheet': 'attention2',
                'order': ['above','within','under'],
                'rename': ['Above', 'Within', 'Under']
                }
        }

        gkey = 'amends'
        # df = pd.read_csv("df.csv", dtype={'cik':object})
        # gkey = 'offer_in_filing_price_range'; df = pd.read_csv("df_revision.csv", dtype={'cik':object})

        attention = df[['underwriter_tier', gkey] + kwkeys]

        kwdict = {
            'IoT_15day_CASI_all': "15 Day CASI (all)",
            'IoT_30day_CASI_all': "30 Day CASI (all)",
            'IoT_60day_CASI_all': "60 Day CASI (all)",
            'ln_15day_CASI_all': "log(CASI) (15 day)",
            'ln_30day_CASI_all': "log(CASI) (30 day)",
            'ln_60day_CASI_all': "log(CASI) (60 day)",
            }

        kwstats = {key:kwtest(key, gkey, df=attention) for key in kwkeys}
        l_stats = {key:l_test(key, gkey, df=attention) for key in kwkeys}
        f_stats = {key:f_test(key, gkey, df=attention) for key in kwkeys}

        Sheet(sumstat_index[gkey]['sheet']).activate()
        Range("A3:T32").value = [['']*44]*44
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






