

import os
import subprocess
import re
import dateutil.parser
import pandas as pd
import progressbar

from itertools      import count
from functools      import wraps
from ftplib         import FTP
from blessings      import Terminal
from progressbar    import ProgressBar, FileTransferSpeed, Percentage, Bar
from concurrent.futures import ProcessPoolExecutor

if float(progressbar.__version__[:3]) > 2.3:
    from progressbar import SimpleProgress
else:
    SimpleProgress = Percentage

N = 6
parse_date = lambda datestr: dateutil.parser.parse(datestr).date()
join_tools = lambda iterable: sum([[x,' '] for x in iterable], [])
TermOffset = lambda y: Terminal().location(0, Terminal().height - y)
FTPWidget  = [' ', Percentage(), Bar(), FileTransferSpeed()]
GrepWidget = [' ', Percentage(), Bar(), SimpleProgress()]
FTPWidget  = join_tools(FTPWidget)
GrepWidget = join_tools(GrepWidget)


class Writer(object):
    """Starts progressbar at specific point on the screen."""
    def __init__(self, y_point):
        """Input: tuple (x, y): position to start progressbar in the terminal.
        Writer((2,5)) -> start progressbar on line 5, 2 spaces in."""
        self.y_point = y_point

    def write(self, string):
        with TermOffset(self.y_point):
            print(string, end=' ')


def GrepProgressbar(enum, df):
    progbar = ProgressBar(fd=Writer(enum+1),
                          maxval=len(df),
                          widgets=GrepWidget)
    return progbar.start()


def parallelize(type):
    "Decorator type: 'threads' or 'processes'."
    def threads_or_processes(f):
        @wraps(f)
        def multiprocess_splice_args(*args, **kwargs):
            """ Preprocesses inputs to create N copies of arguments to map to
            the wrapped function through concurrent.futures.ProcessPoolExecutor.

            Takes only lists of args or dictionaries of keyword args.
            ProcessPoolExecutor cannot unpack args through lambda expressions.
            Must pass a container of inputs: args or kwargs.
            """

            if args:
                file_list = None
                for arg in args:
                    if isFileList(arg):
                        file_list = slice_filelist(arg)
                        args.remove(arg)
                        break

                if not file_list:
                    file_list, *args = args
                    file_list = slice_filelist(file_list)
                    args = sum(args, [])

                static_args = [args]*N
                list_args = list(zip(range(N), file_list, static_args))
                return f(list_args)

            else:
                file_key = [k for k,v in kwargs.items() if isFileList(v)][0]
                list_kwargs = [kwargs.copy() for n in range(N)]
                for n, s in enumerate(slice_filelist(kwargs[file_key])):
                    list_kwargs[n][file_key] = s
                    list_kwargs[n]['enum']   = n
                return f(list_kwargs)

        def isFileList(iterable):
            if isinstance(iterable, pd.DataFrame):
                return True
            if any(map(partial(isinstance,iterable), [pd.Series, set, list, tuple])):
                return True if '.txt' in str(list(iterable)[0]) else False

        def slice_filelist(df):
            "Slices iterable into equally sized iterables"
            if isinstance(df, pd.DataFrame):
                sliced_df = (list(islice(df.values, s, len(df), N)) for s in range(N))
                yield from (pd.DataFrame(x, columns=df.keys()) for x in sliced_df)
            else:
                yield from (list(islice(df, s, len(df), N)) for s in range(N))

        return multiprocess_splice_args
    return threads_or_processes




@parallelize(type='processes')
def fuzzy_match_P(list_of_kwargs):
    """ Multiprocessing wrapper for:

    from fuzzywuzzy import process
    process.extract(query, choices)

    query -> issuer's names from IPOScoop's list.
    choices -> company names from S1 SEC filings

    Example:
        qwer = list(sample['coname'])
        qwer2 = list(sample['gtrends_name'])
        results = fuzzy_match_P(qwer, qwer2)

    args:
        issuers: list of firm names to match
        s1_coname: list of firms names to be matched against.
    """
    print('Starting {} _fuzzy_match() processes...'.format(N))
    with ProcessPoolExecutor(max_workers=N) as exec:
        # execute subprocess:
        results = exec.map(_fuzzy_match, list_of_kwargs)
        flatten = sum(list(results), ())
        hi_match = sum(flatten[::2], [])
        lo_match = sum(flatten[1::2], [])
        return hi_match, lo_match
def _fuzzy_match(args):
    from fuzzywuzzy import fuzz, process
    enum, issuers, s1_coname = args
    alphabet = string.ascii_lowercase
    progbar  = GrepProgressbar(enum+2 , alphabet)
    hi_match   = []
    lo_match   = []

    for i, char in enumerate(alphabet):
        # Narrow down fuzzy match by letter to increase speed
        char_issuers = [x for x in issuers   if x.lower().startswith(char)]
        char_conames = [x for x in s1_coname if x.lower().startswith(char)]
        for issuer in char_issuers:
            with TermOffset(enum+14):
                print("Matching: {}{}".format(issuer, ' '*40))
            match = process.extract(issuer, char_conames)
            m_name, m_score = match[0][0], match[0][1]
            if m_score >= 85:
                hi_match += [(issuer, m_name, m_score)]
            else:
                lo_match += [(issuer, m_name, m_score)]
        progbar.update(i)

    progbar.finish()
    return hi_match, lo_match





# qwer = pd.read_csv("IPOScoop.csv", encoding='latin-1' )
# qwer['dates'] = [aget(d) for d in qwer['dates']]


# aget = lambda datestr: arrow.get(datestr, 'YYYY-MM-DD')
# arng = lambda start, end: arrow.Arrow.range('month', start, end)
# dranges = arng(aget('2005-01-01'), aget('2014-06-01'))

# for start, end in zip(dranges, dranges[1:]):
#     scoopfirms = qwer[(qwer['dates'] > start) & (qwer['dates'] < end)]
#     secfirms =


# df = pd.DataFrame([rawjson[cik]['Company Overview'] for cik in rawjson.keys()])

# for ticker, cname in list(zip(qf['Symbol'], qf['Issuer']))[:4]:

#     # cik = df[df['Proposed Symbol'] == ticker]['CIK'].tolist()[0][3:]
#     scoopfirms = qwer[qwer['Symbol']==ticker]
#     secfirms = df[df['Proposed Symbol'] == ticker]
#     print(ticker, ">>>>>", secfirms['Proposed Symbol'])





def match_IPOscoop_to_SEC():
    # only works when called via main()
    # requires 'fuzzy_match_P' function defined in main.py

    qwer = pd.read_csv("IPOScoop.csv", encoding='latin-1' )

    # qwer['Coname'] = list(map(lambda x: x.lower(), qwer['Issuer']))
    # issuers = qwer.Issuer.tolist()

    SEC_coname = [rawjson[cik]['Company Overview']['Company Name'] for cik in rawjson.keys()]

    hi_match, lo_match = fuzzy_match_P(qwer['Issuer'], SEC_coname)


    firm_lookup = {
        'plains gp holdings lp': [1581990],
        'mead johnson nutrition co': [1452575],
        'renewable energy group inc': [1463258],
        'insys therapeutics inc': [1516479],
        'rexnord corp': [1439288],
        'freightcar america inc': [1320854]
        }

    matchCIKS = []
    for triple in hi_match:
        issuer, m_name, m_score = triple
        m_cik = [x[1] for x in s1_id if x[0] == m_name]
        if len(m_cik) == 1:
            matchCIKS += [[issuer] + m_cik]
        else:
            try:
                matchCIKS += [[issuer] + firm_lookup[m_name]]
            except KeyError:
                print([x for x in s1_id if m_name in  x[0]])


    fdat = pd.DataFrame(matchCIKS, columns=['coname', 'cik'])
    IPO_success = gdat.merge(fdat, how='inner', on=['coname'])
    IPO_success = IPO_success.set_index('cik')
    succ = IPO_success

    new_df = pd.DataFrame(columns=s1.keys())
    for cik, ipo_date in zip(succ.index, succ.Date):
        match = s1[s1.index == cik]
        ipo_date = arrow.get(ipo_date)
        s1_dates = [d for d in match['date'] if arrow.get(d) < ipo_date]
        if len(s1_dates) > 0:
            ref_date = max([arrow.get(d) for d in s1_dates])
            ref_date = str(ref_date.date())
            new_df = new_df.append(match[match['date'] == ref_date])
        else:
            print(cik, ipo_date, s1_dates, '-> No valid S-1 filing date')
            print(match['date'])


    succ = succ.join(new_df, lsuffix='lower')
    succ = succ.drop('conamelower', axis=1)
    succ = succ.dropna()








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
