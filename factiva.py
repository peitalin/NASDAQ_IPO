
import csv
import glob
import pandas as pd
from fuzzywuzzy import process

ffile = 'factiva-weekly/{}.csv'
ffile = ffile.format('2009-08-08')

import requests
import django

import numpy
import pandas
import scipy






def clean_factiva_csv():

	ffiles = glob.glob('factiva-weekly/*.csv')

	def read_factiva(ffile):
		return  open(ffile).read().split('\n\n')[1]

	def df_factiva(ffile):
		raw = read_factiva(ffile)
		ddat = list(csv.reader(raw.splitlines()))
		return  pd.DataFrame(ddat[1:], columns=['Company', 'Count'])

	for ffile in ffiles:
		print("Processing {}".format(ffile))
		df = df_factiva(ffile)
		df.to_csv(ffile, index=False)



def get_filename(ddate):
	yymm = ddate[:-2]
	day = ddate[-2:]

	if int(day) < 8:
		return yymm + '01.csv'
	elif int(day) < 16:
		return yymm + '08.csv'
	elif int(day) < 24:
		return yymm + '16.csv'
	else:
		return yymm + '24.csv'




def news_counts(gtrends_name, ddate):

	try:
		m = pd.read_csv("factiva-weekly/" + get_filename(ddate))
	except:
		print("Error: %s: %s" % (gtrends_name, ddate))
		return 0

	factiva_name, score = process.extractOne(gtrends_name, list(m.Company))
	print("Gtrends: {}\tFactiva: {}\tScore: {}".format((gtrends_name+' '*30)[:35], (factiva_name+' '*30)[:35], score))
	if score > 80:
		count = m[m.Company == factiva_name]['Count'].tolist()[0]
	else:
		count = 0
	return count




def df_media_counts():
	df['media_counts'] = [0] * len(df)
	for gname in df.gtrends_name:
		print(gname, end='\r')
		cik = df[df.gtrends_name == gname].index[0]
		df.loc[cik, 'media_counts'] = media_counts[gname]
	df.to_csv("df.csv", dtype={'cik': object, 'SIC': object, 'Year':object})



if __name__=='__main__':

	df = pd.read_csv("df.csv", dtype={'cik':object, 'Year':object, 'SIC':object})
	df.set_index('cik', inplace=True)

	firmlist = list(zip(list(df.gtrends_name), list(df.date_1st_pricing)))
	firmlist = list(zip(list(df.gtrends_name), list(df.date_trading)))

	news_counts(*firmlist[0])
	media_counts = {f[0]:news_counts(*f) for f in firmlist[:10]}





