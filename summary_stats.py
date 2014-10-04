#!/usr/bin/env python


import os
import re
import sys
import json
import pandas as pd
import numpy as np
import arrow

from widgets import as_cash

from scipy.stats.mstats import kruskalwallis
import seaborn as sb
import matplotlib.pyplot as plt


FILE_PATH = 'text_files/'
IPO_DIR = os.path.join(os.path.expanduser("~"), "Data", "IPO")
BASEDIR = os.path.join(IPO_DIR, "NASDAQ")
FINALJSON = json.loads(open('final_json.txt').read())

aget = lambda x: arrow.get(x, 'M/D/YYYY')
conames_ciks = {cik:FINALJSON[cik]['Company Overview']['Company Name'] for cik in FINALJSON}
firmname = lambda cik: conames_ciks[cik]
get_cik = lambda firm: [x[0] for x in conames_ciks.items() if x[1].lower().startswith(firm)][0]





def lm_model_plots():
    # Set matplotlibrc backend: TkAgg instead of MacOSX

    # ATTENTION
    sb.lmplot("ln_CASI_all_finance", "IPO_duration", sample2,
                hue="underwriter_tier", palette=cp_four("cool_r"),
                robust=True, ci=95, n_boot=500, )

    obs_num = [len(sample[sample['underwriter_tier']==x])
                for x in [ '-1', '0+', '7+', '9']]
    legend_labs = ("No Underwriter, N=",
                   "Rank 0+, N=",
                   "Rank 7+, N=",
                   "Rank 9 (Elites) N=")
    legend_labs = [x+str(y) for x,y in zip(legend_labs, obs_num)]
    plt.legend(legend_labs)
    plt.xlabel(r"$log(CASI)$")
    plt.ylim((-200,1600))
    plt.xlim((-1,11))
    plt.title('Abnomal attention and IPO Duration (bank rank strata)')
    # plt.savefig("IPO_duration_attention.pdf", dpi=200, format='pdf')


def uw_tier_histplots():


    def uw_tier_duration(x):
        return sample[sample.lead_underwriter_tier==x]['IPO_duration']


    from lifelines.estimation import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt

    ranks = ["-1", "0+", "7+", "9"]
    ranklabels = ['No Underwriter', 'Low Rank', 'Mid Rank', 'Rank 9 (elite)']
    kmf = KaplanMeierFitter()
    kwstat = kruskalwallis(*[uw_tier_duration(x) for x in ranks])

    # Success
    f, ax = plt.subplots(1,1,figsize=(12, 4), sharex=True)
    T = 1 # annotation line thickness

    for rank, rlabel, color in zip(ranks, ranklabels, cp_four("cool_r")):
        uw = sample[sample.lead_underwriter_tier==rank]

        kmf.fit(uw['IPO_duration'],
                label='{} N={}'.format(rlabel, len(uw)),
                alpha=0.9)
        kmf.plot(ax=ax, c=color, alpha=0.7)

        quartiles = [int(np.percentile(kmf.durations, x)) for x in [25, 50, 75]][::-1]
        aprops = dict(facecolor=color, width=T, headwidth=T)

        if rank=="-1":
            plt.annotate("75%: {} days".format(quartiles[0]),
                        (quartiles[0], 0.25),
                        xytext=(quartiles[0]+145, 0.25+.04),
                        arrowprops=aprops)

            plt.annotate("50%: {} days".format(quartiles[1]),
                        (quartiles[1], 0.50),
                        xytext=(quartiles[1]+145, 0.50+.04),
                        arrowprops=aprops)

            plt.annotate("25%: {} days".format(quartiles[2]),
                        (quartiles[2], 0.75),
                        xytext=(quartiles[2]+145, 0.75+0.04),
                        arrowprops=aprops)
        elif rank=="9":
            plt.annotate("75%: {} days".format(quartiles[0]),
                        (quartiles[0], 0.25),
                        xytext=(quartiles[0]+415, 0.25+.1),
                        arrowprops=aprops)

            plt.annotate("50%: {} days".format(quartiles[1]),
                        (quartiles[1], 0.50),
                        xytext=(quartiles[1]+290, 0.50+.1),
                        arrowprops=aprops)

            plt.annotate("25%: {} days".format(quartiles[2]),
                        (quartiles[2], 0.75),
                        xytext=(quartiles[2]+165, 0.75+0.1),
                        arrowprops=aprops)

    plt.annotate("Kruskall Wallis\nH: {:.3f}\nprob: {:.3f}".format(*kwstat),
                (960, 0.1))
    plt.ylim(0,1)
    plt.xlim(0,1095)
    plt.title("Kaplan-Meier survival times by bank tier")
    plt.xlabel("IPO Duration (days)")
    plt.ylabel(r"$S(t)=Pr(T>t)$")
    plt.savefig("IPO_tiers_KP_survival.pdf", format='pdf', dpi=200)




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
    kkeys = ['offer_in_filing_price_range', 'percent_first_price_update', 'market_cap', 'percent_final_price_revision',  'log_proceeds', 'share_overhang', 'EPS', 'underwriter_rank_avg', '2month_indust_rets', 'BAA_yield_changes', 'underwriter_tier']

    sample = df[kkeys]
    sample['market_cap'] = sample['market_cap']/1000000 # Mils
    sumstat = sample.groupby(['offer_in_filing_price_range']) \
                    .agg(keystats) \
                    .reindex(index=['above','within','under'])
    sumstat.index = ['Above', 'Within', 'Under']
    # sumstats.to_csv("xls/num_price_updates.csv")
    sumstat

    above, under, within = [x[1] for x in sample.groupby('offer_in_filing_price_range')]


    # sample.groupby(['underwriter_tier']).agg(keystats).to_csv("xls/uw_tier_stats.csv")

    kwtest = lambda s: kruskalwallis(above[s], within[s], under[s])
    kwkeys = ['percent_first_price_update', 'market_cap', 'log_proceeds', 'share_overhang', 'EPS', 'underwriter_rank_avg', '2month_indust_rets', 'BAA_yield_changes']
    kwdict = {
        'percent_first_price_update': 'Percent Price Update',
        'market_cap': 'Market Cap (mil)',
        'log_proceeds': 'ln(Proceeds)',
        'share_overhang': 'Share Overhang',
        'EPS': 'EPS',
        'underwriter_rank_avg': 'Underwriter Rank',
        '3month_indust_rets': 'Industry Returns',
        'BAA_yield_changes': 'BAA Yield Change',
        }
    kwstats = {key:kwtest(key) for key in kwkeys}




    col_names = ['obs', 'mean', 'std', 'min', 'med', 'max', 'Kruskal-Wallis']
    Range("A3:T25").value = [['']*25]*25
    Range("C4").value = col_names
    Range("L4").value = col_names

    cells = sum([['B{}'.format(s), 'K{}'.format(s)] for s in range(5,len(kwkeys)*5,5)], [])    # ['B5', 'B10', 'B15', 'B20', 'K5', 'K10', 'K15', 'K20']

    for i, cell, key in zip(range(8), cells, kwkeys):
        Range(cell).value = sumstat[key]
        Range(cell).value = ['', kwdict[key] ,'','','','','','','']

        kw_cell = next_row(next_col(cell, 7), 3)
        Range(kw_cell).value = 'H-stat'
        Range(next_row(kw_cell)).value = kwstats[key][0]
        Range(next_col(kw_cell)).value = 'p-value'
        Range(next_row(next_col(kw_cell))).value = kwstats[key][1]

        if cell.startswith('K'):
            Range('%s:%s' % (cell, next_row(cell, 4))).value = [['']]*4

        if i != 0:
            Range(next_col(next_row(cell))).value = [['']]*3





if __name__=='__main__':


    df = pd.read_csv('df.csv', dtype={'cik':object})
    df.set_index('cik', inplace=True)
    cik = '1326801' # Facebook


    # amendments = df[~df.size_of_first_price_update.isnull()]
    # revisions = df[~df.size_of_final_price_revision.isnull()]

    # # check .describe() to see key order: above, under, within (alphabetical)
    # above, under, within = [x[1] for x in amendments.groupby('offer_in_filing_price_range')]




    # # Attention plots
    # plotargs = [(success, "Successful IPOs", 5, 1.05, 1.4),
    #             (withdraw, "Withdrawn IPOs", 2, 0.7, 9.1)]
    # plot_var_dist(plotargs, kkey='ln_CASI_all_finance', kw_xy=(7.6, 20))
    # plt.xlabel("Log Cumulative Abnormal Search Interest (all)")
    # plt.title("Abnormal Attention During Book-build IPOs")
    # plt.savefig("./succ_vs_withdraw_CASI.pdf", dpi=200, format='pdf')


    # plot_kaplan_function()
    # plt.savefig("./succ_vs_withdraw_Kaplan_Meier.pdf", dpi=200, format='pdf')
    # uw_tier_histplots()


    # IoT summary stats
    # iot_keys = [x for x in df.keys() if x.startswith('IoT')] + ["offer_in_filing_price_range"]
    # df[iot_keys].groupby("offer_in_filing_price_range").describe()






