

import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import seaborn as sb
import pandas as pd

from scipy.stats.mstats import kruskalwallis
from widgets import as_cash



IPO_DIR = os.path.join(os.path.expanduser("~"), "Data", "IPO")
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ",)
FILEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "Filings")
colors2 = sb.color_palette("husl")


if 'data settings and colors functions':

    def cp_four(x):
        c = sb.color_palette(x, n_colors=16)
        return [c[1], c[7], c[9], c[12]]

    def rgb_to_hex(rgb):
        rgb = map(lambda x: int(max(0, min(x, 255)) * 255), rgb)
        return "#{0:02x}{1:02x}{2:02x}".format(*rgb)

    def set_data(dataframe, groupby_key='percent_final_price_revision', hi=8, lo=0, color_palette='deep'):

        if color_palette == "deep":
            c = sb.color_palette("deep")
            c1, c2, c3 = c[2], c[3], c[5]
        elif color_palette == "husl":
            c = sb.color_palette(color_palette)
            c1, c2, c3 = c[5], c[4], c[3]
        else:
            c = sb.color_palette(color_palette)
            c1, c2, c3 = c[2], c[0], sb.color_palette("husl")[3]

        l, h = round(lo), round(hi)
        g = dataframe


        if 'revision' in groupby_key:
            update = 'Revision'
            gdown = g[g[groupby_key] <= lo]
            gmid  = g[(g[groupby_key] > lo) & (g[groupby_key] < hi)]
            gup   = g[g[groupby_key] >= hi]
            dplotargs = [
                (gdown, c1, "Price {update} <= {l}%".format(update=update, l=l), 0.06),
                (gmid, c2, "{l}% < {update} < {h}%".format(update=update, l=l,h=h), 0.03),
                (gup, c3, "Price {update} >= {h}%".format(update=update, h=h), 0.02)
                ]
        else:
            update = 'Amendment'
            gdown = g[g[groupby_key] < lo]
            gmid  = g[(g[groupby_key] >= lo) & (g[groupby_key] <= hi)]
            gup   = g[g[groupby_key] > hi]
            dplotargs = [
                (gdown, c1, "Price {update} < {l}%".format(update=update, l=l), 0.06),
                (gup, c2, "Price {update} > {h}%".format(update=update, h=h), 0.015)
                ]

        return g, gdown, gmid, gup, dplotargs



def sb_distplots(plotargs, return_key='close_return', update_type='Revisions'):
    "Plots conditional underpricing distributions. Run set_data(df) first."

    f, ax = plt.subplots(1,1,figsize=(16, 5), sharex=True)
    for arg in plotargs:
        df, c, l, h = arg

        sb.distplot(df[return_key], ax=ax,
            kde_kws={"label": l + "    Obs={N}".format(N=len(df)), "color": c},
            hist_kws={"histtype": "stepfilled", "color": c})

        r = df[return_key]
        m,s,y,med = r.mean(), r.std(), r.skew(), r.median()
        ax.annotate(
            u'μ={:.2f}%,   σ={:.2f},   γ={:.2f}'.format(m,s,y),
            xy=(med+2, h), xytext=(med+6, h+0.01),
            arrowprops=dict(facecolor=cl.rgb2hex(c), width=1.5, headwidth=5, shrink=0.1))


    H, prob = kruskalwallis(*[x[0][return_key] for x in plotargs])
    ax.annotate("Kruskal-Wallis: (H={H:.2f}, prob={p:.3f})".format(H=H, p=prob),
                xy=(66,0.01))

    plt.title("Conditional Underpricing Distributions %s" % update_type)
    plt.ylabel("Density")
    plt.xlim(xmin=-40,xmax=100)
    plt.xlabel("1st Day Returns (%)")
    plt.ylim((0, 0.12))













if __name__=='__main__':

    df = pd.read_csv(BASEDIR + '/df.csv', dtype={'cik':object})
    df.set_index('cik', inplace=True)

    amendments = df[~df.size_of_first_price_update.isnull()]
    revisions = df[~df.size_of_final_price_revision.isnull()]

    amendments['percent_final_price_revision'] *= 100
    amendments['percent_first_price_update'] *= 100
    amendments['close_return'] *= 100

    revisions['percent_final_price_revision'] *= 100
    revisions['percent_first_price_update'] *= 100
    revisions['close_return'] *= 100


    # check .describe() to see key order: above, under, within (alphabetical)
    # above, under, within = [x[1] for x in revisions.groupby(['offer_in_filing_price_range'])]


    g, gdown, gmid, gup, dplotargs = set_data(revisions,
                                        groupby_key='percent_final_price_revision',
                                        hi=10, lo=-0.1,
                                        color_palette='deep' )
    sb_distplots(dplotargs, update_type="(Price Revisions)")
    # plt.savefig("conditional-returns-revisions.png", dpi=200)



    g, gdown, gmid, gup, dplotargs = set_data(amendments,
                                        groupby_key='percent_first_price_update',
                                        hi=0, lo=0,
                                        color_palette='colorblind' )
    sb_distplots(dplotargs, update_type="(Price Updates)")
    # plt.savefig("conditional-returns-updates.png", dpi=200)


    ### Linreg plots
    ### Revisions
    """
    c = sb.color_palette("deep")
    c1, c2 , c3 = c[2], c[3], c[5]

    g, gdown, gmid, gup, dplotargs = set_data(revisions,
                                        groupby_key='percent_final_price_revision',
                                        hi=10, lo=-0,
                                        color_palette='deep' )

    sb.jointplot("percent_final_price_revision", "close_return", gdown, kind='reg', color=c1, size=6, ylim=(-100, 200), xlim=(-80, 0))
    plt.savefig("./jointplot-red.png")

    sb.jointplot("percent_final_price_revision", "close_return", gmid, kind='reg', color=c2, size=6, ylim=(-100, 200), xlim=(0,10))
    plt.savefig("./jointplot-purple.png")

    sb.jointplot("percent_final_price_revision", "close_return", gup, kind='reg', color=c3, size=6, ylim=(-100, 200), xlim=(10,100))
    plt.savefig("./jointplot-blue.png")
    """

    ### Updates
    """

    # c = sb.color_palette("husl")
    # c1, c2 , c3 = c[5], c[4], c[3]

    c = sb.color_palette('colorblind')
    c1, c2, c3 = c[2], c[0], sb.color_palette("husl")[3]


    g, gdown, gmid, gup, dplotargs = set_data(revisions,
                                        groupby_key='percent_first_price_update',
                                        hi=0, lo=0,
                                        color_palette='colorblind' )

    sb_distplots(dplotargs, update_type="(Price Updates)")
    # plt.savefig("conditional-returns-updates.png", dpi=200)

    sb.jointplot("percent_first_price_update", "close_return", gdown, kind='reg', color=c1, size=6, ylim=(-50, 200), xlim=(-80, 0))
    plt.savefig("./jointplot-down-updates.png")


    sb.jointplot("percent_first_price_update", "close_return", gup, kind='reg', color=c2, size=6, ylim=(-50, 200), xlim=(0, 80))
    plt.savefig("./jointplot-up-updates.png")
    """


def underwriter_price_update_plots():

    s2 = amendments[['percent_first_price_update' , 'percent_final_price_revision', 'underwriter_tier']]

    s2 = s2[s2['underwriter_tier'] != '-1']
    s2.columns = ['First Price Update (%)','Final Price Revision (%)', 'Rank']


    colors = cp_four("cool_r")
    cool_r = [
            (0.88235294117647056, 0.11764705882352941, 1.0),
            (0.52941176470588236, 0.47058823529411764, 1.0),
            (0.23529411764705888, 0.76470588235294112, 1.0)
            ]

    cblind = [
            sb.color_palette('colorblind')[2],
            sb.color_palette('colorblind')[0],
            sb.color_palette('colorblind')[5],
            ]

    common = [ # red, purple, blue
            sb.color_palette()[2],
            sb.color_palette()[3],
            sb.color_palette()[5],
            ]

    common = [
            (0.15, 0.2870588235294118, 0.4480392156862745),
            sb.color_palette('colorblind')[0],
            (0.9352941176470589, 0.5686274509803922, 0.2),
            ]

    g = sb.PairGrid(s2, hue='Rank',
                        palette=common,
                        size=5)

    g.map_upper(sb.regplot, scatter_kws={"s": 8})
    g.map_lower(plt.scatter, s=10)
    g.map_diag(plt.hist)
    # g.map_upper(sb.kdeplot, cmap="Grey")
    g.set(xlim=(-80, 80), xticks=[-80, -60, -40, -20, 0, 20, 40, 60, 80],
          ylim=(-80, 80), yticks=[-80, -60, -40, -20, 0, 20, 40, 60, 80]);
    g.add_legend()



def histograms_price_update_plot():
    'Plots price-update histograms'

    colors2 = sb.color_palette("husl")

    ###### Price updates
    m, s = amendments['size_of_first_price_update'].mean(), amendments['size_of_first_price_update'].std()
    plt.hist(amendments['size_of_first_price_update'],
                bins=63, alpha=0.6, color=colors2[4], label="First Price Update\n  μ = {:.2f}\n  σ = {:.2f}\n  N = {}".format(m, s, len(amendments)))
    # plt.xticks(xy[1])
    plt.legend()
    plt.xlim(-12,12)
    plt.ylabel("Frequency")
    plt.xlabel("Dollar Price Change ($)")

    ###### Price revisions
    m, s = revisions['size_of_final_price_revision'].mean(), revisions['size_of_final_price_revision'].std()
    plt.hist(revisions['size_of_final_price_revision'],
                bins=48, alpha=0.6,  label="Final Price Revision\n  μ = {:.2f}\n  σ = {:.2f}\n  N = {}".format(m, s, len(revisions)))
    revisions.loc['1117106', 'size_of_final_price_revision'] = 11.5
    # plt.xticks(xy[1])
    plt.legend()
    plt.xlim(-12,12)
    plt.ylabel("Frequency")
    plt.xlabel("Dollar Price Change ($)")


    for cik in df.index:
        if np.isnan(df.loc[cik, 'percent_first_price_update']):
            df.loc[cik, 'pct_first_price_change'] = df.loc[cik, 'percent_final_price_revision']
        else:
            df.loc[cik, 'pct_first_price_change'] = df.loc[cik, 'percent_final_price_revision']


    # xy = sb.lmplot("underwriter_rank_single", "underwriter_rank_avg", data=df)
    # xy.set_xlabels("CM-Rank - Single Top Underwriter")
    # xy.set_ylabels("CM-Rank - Average Lead Underwriters")








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
    sample['Underwriter Tier'] = sample['lead_underwriter_tier']
    sample['IPO Duration'] = sample['IPO_duration']
    ranks = ["-1", "0+", "7+", "9"]

    def uw_tier_duration(x):
        return sample[sample.lead_underwriter_tier==x]['IPO_duration']
    kwstat = kruskalwallis(*[uw_tier_duration(x) for x in ranks])

    # g = sb.FacetGrid(sample,
    #                 row="Underwriter Tier",
    #                 hue="Underwriter Tier",
    #                 palette=cp_four("cool_r"),
    #                 size=2, aspect=4,
    #                 hue_order=ranks, row_order=ranks,
    #                 legend=ranks, xlim=(0,1095))
    # g.map(sb.distplot, "IPO Duration")
    # plt.savefig("IPO_tiers_KP_survival.pdf", format='pdf', dpi=200)


    from lifelines.estimation import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt

    ranks = ["-1", "0+", "7+", "9"]
    ranklabels = ['No Underwriter', 'Low Rank', 'Mid Rank', 'Rank 9 (elite)']
    kmf = KaplanMeierFitter()

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






def plot_var_dist(plotargs, kkey='IPO_duration', kw_xy=(20,20)):

    f, ax = plt.subplots(1,1, figsize=(12, 4), sharex=True)

    for arg in plotargs:
        df, label, color, xshift, yshift = arg
        color = sb.color_palette("muted")[color]
        label += " Obs={}".format(len(df))

        # Summary stats:
        mean = df[kkey].mean()
        mode = df[kkey].mode()
        med  = df[kkey].median()
        std  = df[kkey].std()
        skew = df[kkey].skew()
        stat = u"\nμ={:0.2f}  med={:0.2f}\nσ={:0.2f}  skew={:0.2f}".format(
                mean, med, std, skew)

        yvals, xvals, patchs = plt.hist(df[kkey].tolist(), bins=36, label=label,
                                color=color, alpha=0.6, histtype='stepfilled')

        coords = list(zip(yvals,xvals))
        coords.sort()
        y,x = coords[-3]

        ax.annotate(stat,
                    xy=(x, y),
                    xytext=(x*xshift, y*yshift),
                    arrowprops=dict(facecolor=color,
                                    width=1.6,
                                    headwidth=1.6))

    H, prob = kruskalwallis(*[x[0][kkey] for x in plotargs])
    # U, prob = mannwhitneyu(*[x[0][kkey] for x in plotargs])
    ax.annotate("Kruskal-Wallis: (H={H:.2f}, prob={p:.3f})".format(H=H, p=prob),
                xy=(kw_xy[0], kw_xy[1]))
    plt.ylabel("Frequency")
    plt.legend()


def plot_kaplan_function():

    from lifelines.estimation import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt

    kmf = KaplanMeierFitter()

    # Success
    f, ax = plt.subplots(1,1,figsize=(12, 4), sharex=True)
    T = 1 # annotation line thickness
    kmf.fit(success['IPO_duration'], label='Successful IPOs N=825', alpha=0.9)
    kmf.plot(ax=ax, c=colors[5], alpha=0.7)

    quartiles = [int(np.percentile(kmf.durations, x)) for x in [25, 50, 75]][::-1]
    aprops = dict(facecolor=colors[5], width=T, headwidth=T)

    plt.annotate("75%: {} days".format(quartiles[0]),
                (quartiles[0], 0.25),
                xytext=(quartiles[0]+26, 0.25+.04),
                arrowprops=aprops)

    plt.annotate("50%: {} days".format(quartiles[1]),
                (quartiles[1], 0.50),
                xytext=(quartiles[1]+26, 0.50-.06),
                arrowprops=aprops)

    plt.annotate("25%: {} days".format(quartiles[2]),
                (quartiles[2], 0.75),
                xytext=(quartiles[2]+18, 0.75-0.06),
                arrowprops=aprops)


    # WITHDRAW
    kmf.fit(withdraw['IPO_duration'],
            label = 'Withdrawn IPOs N={}'.format(len(withdraw)),
            event_observed=withdraw['observed'])
    kmf.plot(ax=ax, c=colors[2], alpha=0.7)

    quartiles = [int(np.percentile(kmf.durations, x)) for x in [25, 50, 75]][::-1]
    aprops = dict(facecolor=colors[2], width=T, headwidth=T)

    plt.annotate("75%: {} days".format(quartiles[0]),
                (quartiles[0], 0.25),
                xytext=(quartiles[0]+46, 0.25+.04),
                arrowprops=aprops)

    plt.annotate("50%: {} days".format(quartiles[1]),
                (quartiles[1], 0.50),
                xytext=(quartiles[1]+46, 0.50+.04),
                arrowprops=aprops)

    plt.annotate("25%: {} days".format(quartiles[2]),
                (quartiles[2], 0.75),
                xytext=(quartiles[2]+46, 0.75+.04),
                arrowprops=aprops)

    # log rank tests + general graph labels
    summary, p_value, results = logrank_test(
                                    success['IPO_duration'],
                                    withdraw['IPO_duration'],
                                    alpha=0.95)
    ax.annotate("Log-rank test: (prob={p:.3f})".format(p=p_value),
                xy=(1210, 0.08))

    plt.ylim(0,1)
    plt.xlim(0,1460)
    plt.title("Kaplan-Meier Survival Functions")
    plt.xlabel("IPO Duration (days)")
    plt.ylabel(r"$S(t)=Pr(T>t)$")


    # # Durations plots
    # plotargs = [(success, "Successful IPOs", 5, 1.4, 1.1),
    #             (withdraw, "Withdrawn IPOs", 2, 1.3, 1.2)]
    # plot_var_dist(plotargs, kw_xy=(1100, 10))
    # plt.xlim(xmin=0, xmax=1460)
    # plt.xlabel("IPO Duration (days)")
    # plt.title("Book-build IPO durations")
    # plt.savefig("./succ_vs_withdraw_duration.pdf", dpi=200, format='pdf')


    # # Attention plots
    # plotargs = [(success, "Successful IPOs", 5, 1.05, 1.4),
    #             (withdraw, "Withdrawn IPOs", 2, 0.7, 9.1)]
    # plot_var_dist(plotargs, kkey='ln_CASI_all_finance', kw_xy=(7.6, 20))
    # plt.xlabel("Log Cumulative Abnormal Search Interest (all)")
    # plt.title("Abnormal Attention During Book-build IPOs")
    # plt.savefig("./succ_vs_withdraw_CASI.pdf", dpi=200, format='pdf')


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


