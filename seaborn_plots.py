

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



def cp_four(x):
    c = cp(x, n_colors=16)
    return [c[1], c[7], c[9], c[12]]

def rgb_to_hex(rgb):
    rgb = map(lambda x: int(max(0, min(x, 255)) * 255), rgb)
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)



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


def set_data(dataframe, groupby_key='percent_final_price_revision', hi=8, lo=0, color_palette='deep'):
    g = dataframe
    gdown = g[g[groupby_key] <= lo]
    gmid  = g[(g[groupby_key] > lo) &
              (g[groupby_key]< hi)]
    gup   = g[g[groupby_key] >= hi]

    if color_palette == "deep":
        c = sb.color_palette("deep")
        c1, c2, c3 = c[2], c[3], c[5]
    elif color_palette == "husl":
        c = sb.color_palette(color_palette)
        c1, c2, c3 = c[5], c[4], c[3]
    else:
        c = sb.color_palette(color_palette)
        c1, c2, c3 = c[2], c[0], sb.color_palette("husl")[3]

    l = round(lo)
    h = round(hi)
    if 'revision' in groupby_key:
        update = 'Revision'
    else:
        update = 'Update'
    dplotargs = [
        (gdown, c1, "Price {update} < {l}%".format(update=update, l=l), 0.06),
        (gmid, c2, "{l}% <= {update} < {h}%".format(update=update, l=l,h=h), 0.03),
        (gup, c3, "Price {update} >= {h}%".format(update=update, h=h), 0.02)
        ]
    return g, gdown, gmid, gup, dplotargs



if __name__=='__main__':


    df = pd.read_csv(BASEDIR + '/df.csv', dtype={'cik':object})
    df.set_index('cik', inplace=True)

    ### under, withing, above Filing ranges
    # c = sb.color_palette('colorblind')
    # dplotargs = [
    #     (under, c[2], "Offer Under Filing Price Range", 0.07),
    #     (within, c[0], "Offer Within Filing Price Range", 0.03),
    #     (above, c[5], "Offer Above Filing Price Range", 0.02)
    #     ]
    # sb_distplots(dplotargs, update_type="(Price Revisions)")
    # plt.savefig("conditional-returns-above-within-under.png", dpi=200)

    amendments = df[~df.size_of_first_price_update.isnull()]
    revisions = df[~df.size_of_final_price_revision.isnull()]

    amendments['percent_first_price_update'] *= 100
    revisions['percent_final_price_revision'] *= 100
    revisions['percent_first_price_update'] *= 100
    amendments['close_return'] *= 100
    revisions['close_return'] *= 100

    # check .describe() to see key order: above, under, within (alphabetical)
    above, under, within = [x[1] for x in revisions.groupby(['offer_in_filing_price_range'])]





    g, gdown, gmid, gup, dplotargs = set_data(revisions,
                                        groupby_key='percent_final_price_revision',
                                        hi=10, lo=-0.1,
                                        color_palette='deep' )
    sb_distplots(dplotargs, update_type="(Price Revisions)")
    # plt.savefig("conditional-returns-revisions.png", dpi=200)



    g, gdown, gmid, gup, dplotargs = set_data(amendments,
                                        groupby_key='percent_first_price_update',
                                        hi=10, lo=-0.000,
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


    g, gdown, gmid, gup, dplotargs = set_data(amendments,
                                        groupby_key='percent_first_price_update',
                                        hi=10, lo=-0.000,
                                        color_palette='colorblind' )

    sb_distplots(dplotargs, update_type="(Price Updates)")
    # plt.savefig("conditional-returns-updates.png", dpi=200)

    sb.jointplot("percent_first_price_update", "close_return", gdown, kind='reg', color=c1, size=6, ylim=(-100, 200), xlim=(-80, 0))
    plt.savefig("./jointplot-down-updates.png")

    sb.jointplot("percent_first_price_update", "close_return", gmid, kind='reg', color=c2, size=6, ylim=(-100, 200), xlim=(0, 10))
    plt.savefig("./jointplot-mid-updates.png")

    sb.jointplot("percent_first_price_update", "close_return", gup, kind='reg', color=c3, size=6, ylim=(-100, 200), xlim=(10, 100))
    plt.savefig("./jointplot-up-updates.png")
    """







def price_update_plot():
    'Plots price-update histograms'

    colors2 = sb.color_palette("husl")

    # Histogram for price updates
    # plt.hist(amendments['size_of_first_price_update'],
    #             bins=22, alpha=0.6, color=colors2[4], label="N=%s" % len(amendments))

    amendments.loc['1117106', 'size_of_final_price_revision'] = 12
    plt.hist(amendments['size_of_first_price_update'],
                bins=44, alpha=0.6, color=colors2[4], label="First Price Update: N=%s" % len(amendments))

    plt.hist(amendments['size_of_final_price_revision'],
                bins=48, alpha=0.4, color=colors2[5], label="Final Price Revision: N=%s" % len(amendments))
    amendments.loc['1117106', 'size_of_final_price_revision'] = 11.5
    plt.xticks(xy[1])
    plt.legend()
    plt.xlim(-12,12)
    plt.ylabel("Frequency")
    plt.xlabel("Dollar Price Change($)")



    # Upwards price amendments, and eventual price revision.
    sample3 = revisions[revisions['size_of_first_price_update'] > 0]
    xy = sb.jointplot(
            sample3['percent_first_price_update'],
            sample3['percent_final_price_revision'],
            color=colors2[4]
              # kind='hex'
              )
    xy.set_axis_labels("% First Price Update [Upwards]", "% Final Price Revision")


    # Downwards price amendmets, and eventual price revision.
    sample4 = revisions[revisions.size_of_first_price_update < 0]
    xy = sb.jointplot(
            sample4['percent_first_price_update'],
            sample4['percent_final_price_revision'],
            color=colors2[5]
              # kind='hex'
              )
    xy.set_axis_labels("% First Price Update [Downwards]", "% Final Price Revision")


