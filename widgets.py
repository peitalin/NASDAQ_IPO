

import os
import subprocess
import re
import dateutil.parser
import pandas as pd
import progressbar

from itertools          import count
from functools          import wraps
from ftplib             import FTP
from blessings          import Terminal
from progressbar        import ProgressBar, FileTransferSpeed, Percentage, Bar
from concurrent.futures import ProcessPoolExecutor
from fuzzywuzzy         import fuzz, process

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







def gtrends_names():

    # gdat = pd.DataFrame([FINALJSON[cik]['Company Overview'] for cik in FINALJSON.keys()], FINALJSON.keys())
    # aget = lambda datestr: arrow.get(datestr, 'M/D/YYYY')

    # gdat = gdat[['Company Name', 'Status']]
    # dates = [aget(re.findall(r'\d*/\d*/\d\d\d\d', s)[0]).date() for s in gdat['Status']]
    # gdat['Status'] = [d.strftime('%m-%d-%Y') for d in dates]

    gdat = pd.read_csv("gdat.csv", index_col='cik')
    old_gdat = pd.read_csv('/Users/peitalin/Dropbox/gtrends-beta/cik-ipo/cik-ipos.csv',
                           sep='|', names=['cik', 'coname', 'date'], index_col='cik')

    gdat.index = [str(x) for x in gdat.index]
    old_gdat.index = [str(x) for x in old_gdat.index]
    gset = set(gdat.index)
    gsetold = set(old_gdat.index)

    # list of firms you don't need to gtrend
    # old_gdat.ix[gset & gsetold]

    # list of firms need to scrape
    # ciks2= {cik for cik in gset & gsetold if (a(gdat.ix[cik]['date']) - a(old_gdat.ix[cik]['date'])).days > 30}

    gdat['gname'] = gdat['coname']
    for cik in gset & gsetold:
        print(gdat.loc[cik, 'gname'], "->", old_gdat.loc[cik, 'coname'])
        gdat.loc[cik, 'gname'] = old_gdat.loc[cik, 'coname']

    regexes = [
            r'(( CO[,\.]?)? /[A-Z][A-Z])$',
            r'(s\.a\.a\.)$',
            r'([,]? L[\.\s]?P[\.]?)$',
            r'([,]? INC[\.]?)$',
            r'([,]? LTD[\.]?)$',
            r'([,]? CO[\.])$',
            r'([,]? CORP[\.]?\s*(II)?[I]?)$',
            r'([,]? L[\.]?L[\.]?C[\.]?)$',
            r'([,]? N[\.]?V[\.]?)$',
            r'([,]? S[\.]?A[\.]?)$',
            r'([,]? P[\.]?L[\.]?C[\.]?)$',
            r',$',
            ]

    renamefirms = {
            'Yingli Green Energy Holding CO': 'Yingli Green Energy',
            'Global Education & Technology Group': 'Global Education and Technology Group',
            'Santander Mexico Financial Group, S.a.b. DE C.v.': 'Santander Group',
            'China Techfaith Wireless Communication Technology': 'Techfaith',
            'NEW Oriental Education & Technology Group': 'New Oriental',
            'Country Style Cooking Restaurant Chain': 'Country Style',
            'Home Inns & Hotels Management': 'Home Inns Group',
            'Controladora Vuela Compania DE Aviacion, S.a.b. DE C.v.': 'Volaris',
            'Allied World Assurance CO Holdings, AG': 'Allied World Assurance',
            'Ulta Salon, Cosmetics & Fragrance': 'Ulta Beauty',
            'Alpha Natural Resources, Inc./old': 'Alpha Natural Resources',
            }

    for cik, values in gdat.ix[gset - gsetold].iterrows():
        firm = values['gname']
        for regex in regexes:
            firm = re.sub(regex, '', firm)
        firm = ' '.join(capwords(s) if len(s)>3 else s for s in firm.split(' '))
        # if len(firm) > 4:
        #         print(cik, firm)
        if firm in renamefirms.keys():
            firm = renamefirms[firm]
        gdat.loc[cik, 'gname'] = firm








