

import os, sys
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import seaborn as sb
import pandas as pd
import numpy as np

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

    df.drop(['1087294', '1368308'], inplace=1) # 1st update took longer than a year
    df = df[df.days_to_first_price_change > 0]
    df = df[df.days_to_first_price_change < 300]

    under, above = [x[1] for x in dfu.groupby(['amends'])]

    duration_keys = ["days_from_priced_to_listing",
                    "days_to_final_price_revision",
                    "days_to_first_price_update",
                    "days_from_s1_to_listing",
                    "days_to_first_price_change"]

    amendments = df[~df.size_of_first_price_update.isnull()]
    revisions = df[~df.size_of_final_price_revision.isnull()]

    # amendments['percent_final_price_revision'] *= 100
    # amendments['percent_first_price_update'] *= 100
    # amendments['close_return'] *= 100

    # revisions['percent_final_price_revision'] *= 100
    # revisions['percent_first_price_update'] *= 100
    # revisions['close_return'] *= 100







def conditional_underpricing_plots():
    ### Revisions
    c = sb.color_palette("deep")
    """

    c = sb.color_palette("deep")
    c1, c2 , c3 = c[2], c[3], c[5]
    g, gdown, gmid, gup, dplotargs = set_data(revisions,
                                        groupby_key='percent_final_price_revision',
                                        hi=10, lo=-0,
                                        color_palette='deep' )

    sb_distplots(dplotargs, update_type="(Price Revisions)")
    # plt.savefig("conditional-returns-revisions.png", dpi=200)


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


    g, gdown, gmid, gup, dplotargs = set_data(amendments,
                                        groupby_key='percent_first_price_update',
                                        hi=0, lo=0,
                                        color_palette='colorblind' )
    sb_distplots(dplotargs, update_type="(Price Updates)")
    # plt.savefig("conditional-returns-updates.png", dpi=200)

    sb_distplots(dplotargs, update_type="(Price Updates)")
    # plt.savefig("conditional-returns-updates.png", dpi=200)

    sb.jointplot("percent_first_price_update", "close_return", gdown, kind='reg', color=c1, size=6, ylim=(-50, 200), xlim=(-80, 0))
    plt.savefig("./jointplot-down-updates.png")


    sb.jointplot("percent_first_price_update", "close_return", gup, kind='reg', color=c2, size=6, ylim=(-50, 200), xlim=(0, 80))
    plt.savefig("./jointplot-up-updates.png")
    """


def underwriter_facet_plots():

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


    plt.scatter(revisions.percent_first_price_update,
                revisions.percent_final_price_revision,
                c=common[0], s=12 )
    plt.xlim(-80,80)
    plt.ylim(-80,80)
    plt.xlabel('First Price Update (%)')
    plt.ylabel('Final Price Revision (%)')


    ### Just scatterplot
    r1, r0, r7, r9 = [x[1] for x in revisions.groupby('underwriter_tier')]
    plt.scatter(r0.percent_first_price_update, r0.percent_final_price_revision,
            c=common[0], s=12, label='Rank: 0+')
    plt.scatter(r7.percent_first_price_update, r7.percent_final_price_revision,
            c=common[1], s=12, label='Rank: 7+')
    plt.scatter(r9.percent_first_price_update, r9.percent_final_price_revision,
            c=common[2], s=12, label='Rank: 8.5+')
    plt.xlim(-80,80)
    plt.ylim(-80,80)
    plt.xlabel('First Price Amendment (%)')
    plt.ylabel('Final Price Revision (%)')
    plt.legend()



def histograms_price_update_plot():
    'Plots price-update histograms'

    colors2 = sb.color_palette("husl")
    c = sb.color_palette("deep")

    ###### Price updates
    m, s = amendments['size_of_first_price_update'].mean(), amendments['size_of_first_price_update'].std()
    plt.hist(amendments['size_of_first_price_update'],
                bins=63, alpha=0.6, color=c[3], label="First Price Update\n  μ = {:.2f}\n  σ = {:.2f}\n  N = {}".format(m, s, len(amendments)))
    # plt.xticks(xy[1])
    plt.legend()
    plt.xlim(-10,10)
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


    # for cik in df.index:
    #     if np.isnan(df.loc[cik, 'percent_first_price_update']):
    #         df.loc[cik, 'pct_first_price_change'] = df.loc[cik, 'percent_final_price_revision']
    #     else:
    #         df.loc[cik, 'pct_first_price_change'] = df.loc[cik, 'percent_final_price_revision']



    plt.plot(df.percent_first_price_update, df.percent_final_price_revision)

    # xy = sb.lmplot("underwriter_rank_single", "underwriter_rank_avg", data=df)
    # xy.set_xlabels("CM-Rank - Single Top Underwriter")
    # xy.set_ylabels("CM-Rank - Average Lead Underwriters")








def lm_model_plots():
    # Set matplotlibrc backend: TkAgg instead of MacOSX

    # ATTENTION
    sb.lmplot("ln_CASI_all_finance", "percent_first_price_update",  revisions,
                # hue="amends",
                palette="deep",
                # robust=True,
                # ci=95,
                # n_boot=500,
                )
    # plt.ylim((-50,50))


    # days_from_priced_to_listing
    # days_to_first_price_update
    # percent_first_price_update

    # ATTENTION
    sb.lmplot("ln_CSI_all_finance", "percent_final_price_revision", revisions,
                hue="underwriter_tier", palette="deep",
                # robust=True,
                # ci=95,
                # n_boot=500,
                )
    plt.ylim((-50,50))


    obs_num = {x[0]:len(x[1]) for x in df.groupby('underwriter_tier')}
    legend_labs = {'-1': "No Underwriter: N={}",
                   '0+': "Rank 0+: N={}",
                   '7+': "Rank 7+: N={}",
                   '9' : "Rank 9:  N={}"}

    legend_labs = sorted(legend_labs[k].format(obs_num[k]) for k in obs_num)

    plt.legend(legend_labs)
    plt.xlabel(r"$log(CASI)$")
    plt.ylim((-50,50))
    plt.xlim((-20,20))
    plt.title('Abnormal attention and Price Updates')
    # plt.savefig("CASI_price_updates.pdf", dpi=200, format='png')







def plot_var_dist(plotargs, kkey, kw_xy=(20,20), color="muted"):

    f, ax = plt.subplots(1,1, figsize=(12, 4), sharex=True)
    cpalette = sb.color_palette(color)

    for arg in plotargs:
        df, label, color_num = arg
        color = cpalette[color_num]
        # Summary stats:
        mean = df[kkey].mean()
        med  = df[kkey].median()
        std  = df[kkey].std()
        skew = df[kkey].skew()
        stat = u"\nμ={:0.2f}   med={:0.2f}\nσ={:0.2f}   N={}".format(
                mean, med, std, len(df))
        label += stat

        yvals, xvals, patchs = plt.hist(df[kkey].tolist(), bins=100, label=label,
                                color=color, alpha=0.6, histtype='stepfilled')

    H, prob = kruskalwallis(*[x[0][kkey] for x in plotargs])
    # U, prob = mannwhitneyu(*[x[0][kkey] for x in plotargs])
    ax.annotate("Kruskal-Wallis:\nH={H:.2f}\nprob={p:.3f}".format(H=H, p=prob),
                xy=(kw_xy[0], kw_xy[1]))
    plt.ylabel("Frequency")
    plt.legend()




def plot_kaplan_function(duration_key):

    from lifelines.estimation import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt


    duration_keys = ["days_from_priced_to_listing",
                    "days_to_final_price_revision",
                    # "days_to_first_price_update",
                    "days_from_s1_to_listing",
                    "days_to_first_price_change"]
    duration_key = duration_keys[-1]

    kmf = KaplanMeierFitter()
    f, ax = plt.subplots(1,1,figsize=(12, 4), sharex=True)
    T = 1 # annotation line thickness
    xoffset = 0.4 # annotation offset (x-axis)
    yoffset = 0.04


    # Above filing price range
    kmf.fit(above[duration_key], label='Upward Price Amendment: N={}'.format(len(above)), alpha=0.9)
    kmf.plot(ax=ax, c=colors[5], alpha=0.7)

    quartiles = [int(np.percentile(kmf.durations, x)) for x in [25, 50, 75]][::-1]
    aprops = dict(facecolor=colors[5], width=T, headwidth=T)

    plt.annotate("75%: {} days".format(quartiles[0]),
                (quartiles[0], 0.25),
                xytext=(quartiles[0]+xoffset, 0.25+yoffset),
                arrowprops=aprops)

    plt.annotate("50%: {} days".format(quartiles[1]),
                (quartiles[1], 0.50),
                xytext=(quartiles[1]+xoffset, 0.50+yoffset),
                arrowprops=aprops)

    plt.annotate("25%: {} days".format(quartiles[2]),
                (quartiles[2], 0.75),
                xytext=(quartiles[2]+xoffset, 0.75+yoffset),
                arrowprops=aprops)


    # Under filing price range
    kmf.fit(under[duration_key], label='Downward Price Amendment: N={}'.format(len(under)),)
    kmf.plot(ax=ax, c=colors[2], alpha=0.7)

    quartiles = [int(np.percentile(kmf.durations, x)) for x in [25, 50, 75]][::-1]
    aprops = dict(facecolor=colors[2], width=T, headwidth=T)

    plt.annotate("75%: {} days".format(quartiles[0]),
                (quartiles[0], 0.25),
                xytext=(quartiles[0]+xoffset, 0.25+yoffset+0.05),
                arrowprops=aprops)

    plt.annotate("50%: {} days".format(quartiles[1]),
                (quartiles[1], 0.50),
                xytext=(quartiles[1]+xoffset, 0.50+yoffset+0.05),
                arrowprops=aprops)

    plt.annotate("25%: {} days".format(quartiles[2]),
                (quartiles[2], 0.75),
                xytext=(quartiles[2]+xoffset, 0.75+yoffset+0.05),
                arrowprops=aprops)


    # log rank tests + general graph labels
    # summary, p_value, results = logrank_test(
    #                                 above[duration_key],
    #                                 within[duration_key],
    #                                 under[duration_key],
    #                                 alpha=0.95)
    # ax.annotate("Log-rank test: (prob={p:.3f})".format(p=p_value),
    #             xy=(1210, 0.08))

    plt.ylim(0,1)
    plt.xlim(0, max(np.percentile(above[duration_key], 90), np.percentile(under[duration_key],90)))
    plt.title("Kaplan-Meier Survival Functions")
    plt.xlabel("Delay (days) in {}".format(duration_key))
    plt.ylabel(r"$S(t)=Pr(T>t)$")






"""

# # Attention plots
## REVISIONS
########


days = 15
IOTKEY = 'IoT_{}day_CASI_weighted_finance'.format(days)
above, under, within = [x[1] for x in revisions.groupby('offer_in_filing_price_range')]

cik1, cik2 = above[above[IOTKEY] == 0][:2].index
cik3, cik4 = under[under[IOTKEY] == 0][:2].index
cik5, cik6 = within[within[IOTKEY] == 0][:2].index
above.loc[cik1, IOTKEY] = 20
above.loc[cik2, IOTKEY] = -20
under.loc[cik3, IOTKEY] = 20
under.loc[cik4, IOTKEY] = -20
within.loc[cik5, IOTKEY] = 20
within.loc[cik6, IOTKEY] = -20
########

plotargs = [
        (within, "Within Price Range", 3),
        (under, "Under Price Range", 2),
        (above, "Above Price Range", 5),
            ]
plot_var_dist(plotargs, kkey=IOTKEY, kw_xy=(-18, 12))
plt.xlabel(IOTKEY.replace("_", " "))
plt.title("Abnormal Attention and Price Revision Groups")
######
above.loc[cik1, IOTKEY] = 0
above.loc[cik2, IOTKEY] = 0
under.loc[cik3, IOTKEY] = 0
under.loc[cik4, IOTKEY] = 0
within.loc[cik5, IOTKEY] = 0
within.loc[cik6, IOTKEY] = 0
#####
plt.savefig("/Users/peitalin/Desktop/iot_revisions_{}d.png".format(days), dpi=200, format='png')




## UPDATES


days = 60
IOTKEY = 'IoT_{}day_CASI_weighted_finance'.format(days)
down, noupdate, up = [x[1] for x in revisions.groupby('amends')]
cik1, cik2 = above[above[IOTKEY] == 0][:2].index
cik3, cik4 = under[under[IOTKEY] == 0][:2].index
cik5, cik6 = within[within[IOTKEY] == 0][:2].index
down.loc[cik1, IOTKEY] = 20
down.loc[cik2, IOTKEY] = -20
up.loc[cik3, IOTKEY] = 20
up.loc[cik4, IOTKEY] = -20
noupdate.loc[cik5, IOTKEY] = 20
noupdate.loc[cik6, IOTKEY] = -20

plotargs = [
            (noupdate, "No Update", 3),
            (down, "Down", 2),
            (up, "Up", 5),
            ]
plot_var_dist(plotargs, kkey=IOTKEY, kw_xy=(-12, 12))
plt.xlabel(IOTKEY.replace("_", " "))
plt.title("Abnormal Attention and Price Amendments")
#####
down.loc[cik1, IOTKEY] = 0
down.loc[cik2, IOTKEY] = 0
up.loc[cik3, IOTKEY] = 0
up.loc[cik4, IOTKEY] = 0
noupdate.loc[cik5, IOTKEY] = 0
noupdate.loc[cik6, IOTKEY] = 0
#####
plt.savefig("/Users/peitalin/Desktop/iot_amends_{}d.png".format(days), dpi=200, format='png')


"""



def uw_tier(uw_rank):
    if uw_rank > 8.5:
        return "8.5+"
    elif uw_rank > 7:
        return "7+"
    elif uw_rank >= 0:
        return "0+"
    elif uw_rank < 0:
        return "-1"


"""


df = pd.read_csv(BASEDIR + '/df.csv', dtype={'cik':object})
df.set_index('cik', inplace=True)


df['underwriter_tier'] = [uw_tier(r) for r in df['underwriter_rank_avg']]
amendments = df[~df.size_of_first_price_update.isnull()]
revisions = df[~df.size_of_final_price_revision.isnull()]

amendments['percent_final_price_revision'] *= 100
amendments['percent_first_price_update'] *= 100
amendments['close_return'] *= 100

revisions['percent_final_price_revision'] *= 100
revisions['percent_first_price_update'] *= 100
revisions['close_return'] *= 100
r1, r0, r7, r9 = [x[1] for x in revisions.groupby('underwriter_tier')]


days = 15
IOTKEY = 'IoT_{}day_CASI_weighted_finance'.format(days)
plotargs = [
            (r9, "Rank 8.5+", 2),
            (r7, "Rank 7+", 5),
            (r0, "Rank 0+", 0)
            ]
plot_var_dist(plotargs, kkey='IoT_{}day_CASI_all'.format(days), kw_xy=(-12, 22), color='colorblind')

plt.xlabel(IOTKEY.replace("_", " "))
plt.title("Abnormal Attention and Underwriter Rank")

plt.savefig("/Users/peitalin/Desktop/iot_underwriter_{}d.png".format(days), dpi=200, format='png')

"""





"""



def abnormal_svi(df, window=15, category='all'):
    ASVI = pd.read_csv("IoT/ASVI_{}day_{}.csv".format(window, category), dtype={'cik': object}).set_index('cik')
    return ASVI

common = [
        (0.15, 0.2870588235294118, 0.4480392156862745),
        sb.color_palette('colorblind')[0],
        (0.9352941176470589, 0.5686274509803922, 0.2),
        ]
ASVI15 = abnormal_svi(df, window=15, category='weighted_finance')
ASVI30 = abnormal_svi(df, window=30, category='weighted_finance')
ASVI60 = abnormal_svi(df, window=60, category='weighted_finance')

r1, r0, r7, r9 = [v[1].index for v in df.groupby("underwriter_tier")]


fig, ax  = plt.subplots(3)

#  15 day
ax[0].plot(range(-15,16), ASVI15.loc[r0].mean(), label='Rank 0+', c=common[0])
ax[0].plot(range(-15,16), ASVI15.loc[r7].mean(), label='Rank 7+', c=common[1])
ax[0].plot(range(-15,16), ASVI15.loc[r9].mean(), label='Rank 8.5+', c=common[2])
ax[0].legend()

#  30 day
ax[1].plot(range(-30,31), ASVI30.loc[r0].mean(), label='Rank 0+', c=common[0])
ax[1].plot(range(-30,31), ASVI30.loc[r7].mean(), label='Rank 7+', c=common[1])
ax[1].plot(range(-30,31), ASVI30.loc[r9].mean(), label='Rank 8.5+', c=common[2])


#  60 day
ax[2].plot(range(-60,61), ASVI60.loc[r0].mean(), label='Rank 0+', c=common[0])
ax[2].plot(range(-60,61), ASVI60.loc[r7].mean(), label='Rank 7+', c=common[1])
ax[2].plot(range(-60,61), ASVI60.loc[r9].mean(), label='Rank 8.5+', c=common[2])

plt.xlabel("Days before and after final price revision")
plt.suptitle("CASI Before and After Price Revision", size=12)
plt.savefig("/Users/peitalin/Desktop/CASI_uw.png", dpi=200, format='png')



"""

