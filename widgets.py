

import os
import subprocess
import json
import re
import glob
import dateutil.parser
import pandas as pd


from functools          import wraps
from concurrent.futures import ProcessPoolExecutor
from fuzzywuzzy         import fuzz, process



if '__price_range_parsing_functions':

    def fix_bad_str(string):
        """ Formats nuisance characters. """
        if string:
            # These are various long dash characters used in the document
            for r in [r'\x97', r'\x96', r'[-]+', r'●']:
                string = re.sub(r, '-', string)
            # Other nuisance chars
            string = re.sub(r'\x95', '->', string)
            string = re.sub(r'\x93', '"', string)
            string = re.sub(r'\x94', '"', string)
            string = re.sub(r'/s/', '', string)
            string = re.sub(r'\x92', "'", string)
            string = re.sub(r'\xa0', ' ', string)
            string = re.sub(r'\u200b', '', string)
            string = re.sub(r'\s+', ' ', string)
        return string.strip()

    def fix_dollars(string_list):
        """Split and strip a string, appending dollar signs where necessary"""

        new_strlist = []
        prepend_next = False
        for s in string_list:
            s = re.sub(r'^(U[\.]?S[\.]?)?\$\s*', '$', s)
            if s.strip() == '$':
                prepend_next = True
                continue
            new_str = ' '.join(e.strip() for e in s.split('\n'))
            if prepend_next == True:
                if not new_str.startswith('$'):
                    new_str = '$' + new_str
                prepend_next = False

            new_strlist += [new_str]
        return new_strlist


    def as_cash(string):
        if '$' not in string:
            return None
        string = string.replace('$','').replace(',','')
        return float(string) if string.strip() else None


    def view_filing(filename):
        FILEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ", "filings")
        filename = filename.split('/')[-2:]
        filename = os.path.join(*[FILEDIR] + filename)
        newfilename = '/Users/peitalin/Public/' + filename.split('/')[-1] + '.html'
        os.system("cp {0} {1}".format(filename, newfilename))
        os.system("open -a Firefox {}".format(newfilename))



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



def write_FINALJSON(FINALJSON):
    "Backup FINALJSON.txt (final_json@@@@.txt) and save current FINALJSON json."
    import shutil, glob
    BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ")
    src_file = os.path.join(BASEDIR, 'final_json.txt')
    dest_file = glob.glob(os.path.join(BASEDIR, 'data') + \
                 '/final_json@*')[-1].replace('@.', '@@.')
    shutil.copy(src_file, dest_file)
    print("Moving {} to {}".format(src_file, dest_file))
    print("Saving current FINALJSON dictionary as final_json.txt")
    with open('final_json.txt', 'w') as f:
        f.write(json.dumps(FINALJSON, indent=4, sort_keys=True))




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


    for cik in newciks:
        firm = FINALJSON[cik]['Company Overview']['Company Name']
        for regex in regexes:
            firm = re.sub(regex, '', firm)
        firm = ' '.join(capwords(s) if len(s)>3 else s for s in firm.split(' '))
        print(firm)
        gdat.loc[cik, 'gname'] = firm
        gdat.loc[cik, 'cik'] = cik
        gdat.loc[cik, 'coname'] = firmname(cik)
        m, d, y = [''.join(s for s in string if s.isdigit())
                            for string in company.loc[cik, 'Status'].split('/')]
        m = '0' + m if len(m) == 1 else m
        d = '0' + d if len(d) == 1 else d
        gdat.loc[cik, 'date'] = '{m}-{d}-{y}'.format(m=m, d=d, y=y)



def delete_old_gtrends(L, remove=False):
    "Finds gtrends files with less than L lines and delets them"

    for ffile in glob(gtdir+'*/*.csv'):
        lines = sum(1 for line in open(ffile))
        if lines < L:
            cik = ffile[-11:-4]
            if remove:
                os.remove(ffile)
            try:
                ipo_date = company.loc[cik, 'Status']
                print("{}: {} <{}> lines => {}".format(cik, firmname(cik), ipo_date, lines))
            except KeyError:
                pass
    print("\n Update these files from Gtrends. ")




# ['0'+str(i) if len(str(i))<7 else str(i) for i in gdat.cik]


# newnames = {'1580608': "Santander Consumer USA", '1552275': "Susser Petroleum", '1478242': "Quintiles Transnational", '1230276': "Pandora", '1359055': "Buckeye Partners LP", '1103025': "TRX Inc", '1357371': "Breitburn Energy", '1403795': "Nivs Intellimedia", '1524931': "Chuy’s", '1297627': "Spansion", '1364541': "Eagle Rock Energy", '1549848': "Hi Crush Partners", '1415301': "DEL Frisco's Restaurant", '1259515': "Control4", '1362120': "Pinnacle Gas", '1603145': "NextEra Energy Resources", '1286613': "Lincoln Educational", '1601072': "Abengoa", '1586049': "Oxford Immunotec", '1162192': "Avalon Pharma", '1004724': "AdCare Health", '1490165': "Erickson Inc", '1289419': "Morningstar Inc", '1316175': "Anthera Pharmaceuticals", '0054003': "Jorgensen Earle", '1545391': "TCP Lighting", '1526160': "Fleetmatics", '1576044': "QEP Midstream", '1428669': "SolarWinds Inc", '1145197': "Insulet", '1325702': "Magnachip", '1317362': "Shamir Lens", '1403431': "Heritage Crystal Clean", '1437260': "Navios Maritime Holdings", '1324272': "Ruth’s Chris Steak House", '1392091': "SemGroup Energy", '1602367': "First Choice Emergency Room", '1578453': "Dynagas LNG", '1547638': "Stanley Inc", '1487101': "KEYW Corporation", '1405419': "Silver Airways", '1314822': "SES World Skies", '1337117': "Ituran Location and Control", '1408100': "Ion Media Networks", '1441634': "Avago Technologies", '1411579': "AMC Theatres", '1574111': "Propensa", '1367064': "Exterran Holdings", '1302028': "Manitex", '1574596': "The New Home Company", '1383650': "Cheniere Energy Inc", '1428875': "Servicemaster", '1018979': "Amerisafe", '1385544': "Horsehead Corporation", '1356949': "Houston Wire and Cable", '1405197': "Talecris Biotherapeutics", '1401688': "Vitacost", '1122388': "Ellie Mae", '1371782': "MV Oil Trust", '1285701': "MMRGlobal", '1407463': "Pioneer Southwest Energy", '1371455': "Evraz Claymont Steel", '1385613': "Greenlight Capital", '1310114': "ServiceSource", '1337675': "Jazz Semiconductor", '1326200': "Genco ATC", '1585854': "SunEdison", '1341769': "Grubb and Ellis Realty", '1325670': "Sonabank", '0748015': "Sealy Corporation", '1513761': "Norwegian Cruise Line", '1426945': "Echo Global Logistics", '1324410': "Guaranty Bank", '1449488': "compressco", '1402436': "SS&C Technologies", '1555538': "Suncoke Energy", '1335106': "China Shenghuo Pharmaceutical", '1142576': "Optimer Pharmaceuticals", '1125001': "Iomai", '1535929': "Voya Financial", '0890264': "Exa Corporation", '1575828': "Frank’s International", '1354217': "Volcano Corporation", '1421150': "Britannia Bulk", '1552797': "Delek Inc", '1097503': "Navteq", '1246263': "Magellan Midstrean", '1515673': "Ultragenyx", '1381074': "Fuwei Films", '1394074': "Spectra Energy", '1310897': "PanAmSat", '1392380': "Gevo Inc", '1312928': "Kayak Corporation", '1362705': "Constellation Energy", '1582966': "Cheniere Energy", '1584831': "Oxbridge Re", '1602065': "Viper Energy", '1410838': "El Paso Pipeline", '1578318': "Envision Healthcare", '1338613': "Regency Energy", '1411583': "Williams Pipeline", '1237746': "Endurance International Group", '1564180': "Knot Offshore", '1605725': "Vtti Energy", '1604665': "Westlake Chemical Corporation", '0923144': 'Williams Scotsman International', '1230355': 'Baxano Surgical'}


# for cik in newnames:
#     df.loc[cik, 'gtrends_name'] = newnames[cik]


# for cik in rmciks:
#     rmfiles = glob.glob('/Users/peitalin/Dropbox/gtrends-beta/cik-ipo/*/{}.csv'.format(cik))
#     for ffile in rmfiles:
#         print(ffile)
#         os.remove(ffile)

