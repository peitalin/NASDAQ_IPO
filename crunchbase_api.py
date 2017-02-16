

import requests
import json
import time
from fuzzywuzzy import process, fuzz


API_KEY=''


class EntityClient(object):

	def __init__(self, user_key=API_KEY):
		self.user_key = user_key
		self.base_url = 'http://api.crunchbase.com/v/2'

	def _format(self, response, format):
		if format == 'json':
			return response.json()
		elif format == 'text':
			return response.text
		elif format == 'response':
			return response

	def people(self, category, page=1, order='created_at+DESC', format='json'):
		"""Args:
		order = ['created_at+DESC', 'created_at+ASC', 'updated_at+DESC', 'update_at+DESC']
		categories = ['organizations', 'people', 'products', 'locations', 'categories']
		"""
		url = '{base_url}/{category}?user_key={user_key}&page={page}&order={order}'.format(
			base_url=self.base_url,
			category=category,
			user_key=self.user_key,
			page=page,
			order=order)

		return self._format(requests.get(url), format)


	def get_uuid(self, permalink, format='json'):
		url = "{base_url}/organization/{permalink}?user_key={user_key}".format(
			base_url=self.base_url,
			permalink=permalink,
			user_key=self.user_key)
		self.uuid = self._format(requests.get(url), format)['data']['uuid']

		return self.uuid


	def organization(self, permalink, format='json'):
		url = "{base_url}/organization/{permalink}?user_key={user_key}".format(
			base_url=self.base_url,
			permalink=permalink,
			user_key=self.user_key)

		return self._format(requests.get(url), format)


	def IPO(self, uuid, format='json'):

		if not uuid:
			uuid = self.get_uuid(self.permalink)

		url = '{base_url}/ipo/{uuid}?user_key={user_key}'.format(
			base_url=self.base_url,
			user_key=self.user_key,
			uuid=uuid)

		return self._format(requests.get(url), format)













def listCompanyInvestors(company):
    """Returns a list of financial organizations
    invested in a given company"""
    investors = set()
    for rounds in company['funding_rounds']:
        for org in rounds['investments']:
            if org['financial_org'] is not None:
                investors.add(org['financial_org']['name'])
    return list(investors)

def listInvestorPortfolio(investor):
    """Returns a list of companies invested in by orgName"""
    portfolio = set()
    for investment in investor['investments']:
        portfolio.add(investment['funding_round']['company']['name'])
    return list(portfolio)







CRUNCHBASEKEY = 'b9ac4c63db75a77c4ea3401d5614c8a8'
uuid = 'df6628127f970b439d3e12f64f504fbb'




if __name__=='__main__':

    EC = EntityClient(CRUNCHBASEKEY)
    pdat = pd.read_csv("permalinks.csv")
    pdat.set_index('cik', inplace=True)

    for cik, permalink in pdat['permalink'].items():
        uuid = pdat.loc[cik, 'uuid']
        if uuid == uuid:
            continue

        print('{}: Retreiving permalink: <{}>'.format(cik, permalink), end=' '*30+'\r')
        try:
            pdat.loc[cik, 'uuid'] = EC.get_uuid(permalink)
            time.sleep(2)
        except KeyboardInterrupt:
            break
        except:
            print('{}: JSONDecodeError: <{}>{}'.format(cik, permalink, ' '*30))


    # df = pd.read_csv("df.csv", dtype={'cik': object})
    # df.set_index("cik", inplace=True)
















## VC funding
def crunchbase_api():

    client = CrunchBaseClient(CRUNCHBASEKEY)

    # record = client.companies.get_by_name('dropbox')
    # kkeys  = record.keys()
    # record['funding_rounds']
    # dropbox_VC = listCompanyInvestors(record)

    elon_musk = client.people.get_by_name('Elon Musk')
    listInvestorPortfolio(elon_musk)

    all_companies = client.companies.get_all()
    all_people    = client.people.get_all()
    all_finorgs   = client.financial_organizations.get_all()

    with open("data/all_companies.txt", 'w') as f:
        json.dump(all_companies, f)

    with open("data/all_people.txt", 'w') as f:
        json.dump(all_people, f)

    with open("data/all_finorgs.txt", 'w') as f:
        json.dump(all_finorgs, f)

    return all_companies, all_people, all_finorgs


def get_crunchbase(entity="companies", dataframe=True):
    "entity = companies, people or finorg"
    if not dataframe:
        with open("data/all_{}.txt".format(entity), 'r') as f:
            return json.loads(f.read())
    else:
        return pd.read_json("data/all_{}.txt".format(entity))



# def get_permalinks():
#      Crunchbase API Version 1 # client = CrunchBaseClient('w6zh9bmr8h3cwnekeycyyzh2')
#     record = client.companies.get_by_name('dropbox')
#     kkeys  = record.keys()
#     record['funding_rounds']
#     dropbox_VC = listCompanyInvestors(record)

#     elon_musk = client.people.get_by_name('Elon Musk')
#     listInvestorPortfolio(elon_musk)

#     all_companies = get_crunchbase()
#     all_companies.set_index('name', inplace=True)

#     permalinks = {}
#     for cik, coname, gt_name in df[['Coname', 'gtrends_name']].itertuples():
#         alpha = gt_name[:5].lower()
#         firmlist = [x for x in all_companies.index if x.lower().startswith(alpha)]
#         match = process.extractOne(gt_name, firmlist)
#         if not match:
#             continue
#         if match[1] < 88:
#             continue
#         if match[1] < 94:
#             print("{} => {}".format(gt_name, match))
#             keep = input("keep? <y> or skip? <ENTER>: ")
#             if keep.lower() == 'y':
#                 permalinks.update({cik:all_companies.loc[match[0], 'permalink']})
#             else:
#                 print("skipped\n")
#                 continue
#         permalinks.update({cik:all_companies.loc[match[0], 'permalink']})

    # inc_dict = {
    #     '0018169': 'dole-food-company',
    #     '1539838': 'diamondback-partners',
    #     '1281774': 'town-sports-international-new-york-sports-clubs',
    #     '1375829': 'rrsat',
    #     '1579877': 'cbs-outdoor',
    #     }

    # with open("crunch_permalinks.txt", 'w+') as f:
    #     f.write(json.dumps(permalinks, indent=4, sort_keys=True))


