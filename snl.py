
import requests
import os
import time

from subprocess import Popen
from lxml import html

from selenium import webdriver
from selenium.webdriver.common import action_chains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import *


URL="http://www.snl.com/InteractiveX/default.aspx"
USERNAME="huong.truong@uqconnect.edu.au"
PASSWORD="J99hg3k6"
df = pd.read_csv("snl_document_list.csv", dtype={"Document ID": object})
df.set_index("Document ID", inplace=True)
Documents = []
global d


def initialize_SNL_login(USERNAME=USERNAME, PASSWORD=PASSWORD, URL=URL):
    "Requires Firefox Browser to login to snl.com"

    # initialize Firefox Webbrowser
    d = webdriver.Firefox()
    # d = webdriver.PhantomJS()
    d.get(URL)
    # find login assets and submit login details
    username_login = d.find_element_by_name("username")
    password_login = d.find_element_by_name("password")
    username_login.send_keys(USERNAME)
    password_login.send_keys(PASSWORD)

    # check this is the correct sign-in button
    if d.find_element_by_link_text("Sign In") == d.find_element_by_class_name("submit"):
        d.find_element_by_link_text("Sign In").click()

    print("Login success!")
    return d



def set_query_dates(begin_date='1/1/2016', end_date=None):

    # inject javascript code to reveal hidden documents settings toggle
    toggle_settings = d.find_element_by_xpath(
            '//td[@id="settingsHeader_ctl00_ctl06_SNLSettingsBox4"]')
    if not toggle_settings.is_displayed():
        jscript = "SNLSettingsBox_ctl00_ctl06_SNLSettingsBox4.ToggleSettingsBox()"
        d.execute_script(jscript)

    # check that "Issued Date" checkbox is correct item
    if begin_date:
        TextBox = d.find_element_by_xpath('//input[@id="ctl00_ctl06_tbBeginDate"]')
        TextBox.clear()
        TextBox.send_keys(begin_date)

    if end_date:
        TextBox = d.find_element_by_xpath('//input[@id="ctl00_ctl06_tbEndDate"]')
        TextBox.clear()
        TextBox.send_keys(end_date)

    time.sleep(0.3)
    # click on empty space to close pop-up menu blocking "apply" button
    d.find_element_by_xpath('//div[@class="borderToolBox"]').click()
    # find "apply" button and click it
    d.find_element_by_id("Apply_ctl00_ctl06_SNLSettingsBox4").click()


def get_Doc_details(page_source):

    hh = html.fromstring(page_source)
    Document = {}

    try:
        xpath_ ='//tr[contains(@id, "ctl00_ctl06_m_FilingTypeRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text
        dval = hh.xpath(xpath_ + '/td/a')[0].text
        Document[dkey] = dval

        xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_CompanyRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text

        dval = hh.xpath(xpath_ + '/td/a')[0].text
        Document[dkey] = dval

        xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_EventDateRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text
        dval = hh.xpath(xpath_ + '/td')[1].text
        Document[dkey] = dval

        try:
            xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_FilingDateRow")]'
            dkey = hh.xpath(xpath_ + '/td')[0].text
            dval = hh.xpath(xpath_ + '/td')[1].text
            Document[dkey] = dval
        except IndexError:
            pass

        xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_DocumentIDRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text
        dval = hh.xpath(xpath_ + '/td')[1].text
        Document[dkey] = dval

        try:
            xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_AbstractRow")]'
            dkey = hh.xpath(xpath_ + '/td')[0].text
            dval = hh.xpath(xpath_ + '/td')[1].text
            Document[dkey] = dval
        except IndexError:
            xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_RelatedDealsRow")]'
            dkey = hh.xpath(xpath_ + '/td')[0].text
            dval =  " ".join(a.text for a in hh.xpath(xpath_ + '/td/a'))
            Document[dkey] = dval
        try:
            xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_RelatedArticlesRow")]'
            dkey = hh.xpath(xpath_ + '/td')[0].text
            dval = " ".join(a.text for a in hh.xpath(xpath_ + '/td/a'))
            Document[dkey] = dval
        except IndexError:
            pass


    except IndexError:

        xpath_ ='//tr[contains(@id, "ctl00_ctl06_m_FilingTypeRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text
        dval = hh.xpath(xpath_ + '/td/a')[0].text
        Document[dkey] = dval

        xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_PresentationTitleRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text
        dval = hh.xpath(xpath_ + '/td')[1].text
        Document[dkey] = dval


        xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_PresentationDateRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text
        dval = hh.xpath(xpath_ + '/td')[1].text
        Document[dkey] = dval

        xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_PresentersRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text
        dval = hh.xpath(xpath_ + '/td/a')[0].text
        Document[dkey] = dval

        xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_DocumentIDRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text
        dval = hh.xpath(xpath_ + '/td')[1].text
        Document[dkey] = dval

        xpath_ = '//tr[contains(@id, "ctl00_ctl06_m_AbstractRow")]'
        dkey = hh.xpath(xpath_ + '/td')[0].text
        dval = hh.xpath(xpath_ + '/td')[1].text
        Document[dkey] = dval

    return {k:v.replace('\xa0', ' ') for k,v in Document.items()}






if __name__=='__main__':
    print(1)


d = driver = initialize_SNL_login()
# After login, navigate to 'Industries/Enforcement Actions' tab
industries_tab = "/SNLWebPlatform/Content/Industry/IndustryAnalysis.aspx"
time.sleep(0.1)
d.find_element_by_xpath('//a[@href="' + industries_tab + '"]').click()
d.find_element_by_link_text("Enforcement Actions").click()
time.sleep(0.1)


# javascript generated dom, invisible until we inject javascript to reveal element
# set_query_dates('1/1/2015')
try:
    set_query_dates(begin_date, end_date)
except NameError:
    begin_date = input("Begin date (MM/DD/YYYY): ")
    end_date = input("End date (MM/DD/YYYY): ")
    set_query_dates(begin_date, end_date)

print("Set Date Range: {} ~ {}".format(begin_date, end_date))
# end date by default is current date

# change from pagination to "View All" for enforcement listings
print("Loading all filings from pagination...")
d.execute_script("PaginateOwnershipGrid('0')")
time.sleep(0.2)




# list of all filings in table:
table = d.find_element_by_xpath('//table[@id="ctl00_ctl06_enforcementActionGrid"]')
alist = table.find_elements_by_xpath('//a[contains(@href, "Doc.aspx")]')

for i, a in enumerate(alist):

    try:
        try:
            pdf_page_url = a.get_attribute('href')
        except:
            time.sleep(0.2)
            pdf_page_url = a.get_attribute('href')

        if pdf_page_url.split('=')[-1] in df.index:
            print("{}.pdf exists, skipping".format(pdf_page_url.split('=')[-1]))
            continue

        # open new tab and load page_source
        a.send_keys(Keys.COMMAND + 't')
        d.get(pdf_page_url)
        # extract document details from page_source
        Doc = get_Doc_details(d.page_source)


        # load pdf file generated by jscript from page_source, can't scrape link directly
        pdf_link = d.find_element_by_xpath('//a[contains(@href, "KeyFileFormat=PDF")]')
        pdf_link.click()
        # pdf_link = d.find_element_by_xpath('//a[contains(@href, "KeyFileFormat=XML")]')
        # pdf_link.click()


        # use wget to download file
        current_url = d.current_url
        file_name   = "{}.pdf".format(Doc['Document ID'])

        if not os.path.exists(file_name):
            cmd_exec = "wget -O {} {}".format(file_name, current_url)
            Popen(cmd_exec, shell=True).wait()

        Documents.append(Doc)

        # close tab
        d.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 'w')
        print("\n[FILING OBTAINED] >>>>> {} <<<<<".format(pdf_page_url))
        print("[{}/{}]\n".format(i+1, len(alist)))
        time.sleep(0.1) # must set delay: 0.1~0.3 seconds
        # waits for tab to close and refocus driver on main window.
		# in firefox: type about:config -> animate -> disable tab animation

    # except (IndexError, StaleElementReferenceException, NoSuchElementException):
    except:

        df = pd.read_csv("snl_document_list.csv", dtype={"Document ID": object})
        df.set_index("Document ID", inplace=True)

        df2 = pd.DataFrame(Documents)
        df2.set_index("Document ID", inplace=True)

        df = df.append(df2).drop_duplicates()
        df.to_csv("snl_document_list.csv", dtype={"Document ID": object})

        d.find_element_by_tag_name('body').send_keys(Keys.COMMAND + 'w')

        pass



df = pd.read_csv("snl_document_list.csv", dtype={"Document ID": object})
df.set_index("Document ID", inplace=True)

df2 = pd.DataFrame(Documents)
df2.set_index("Document ID", inplace=True)

df = df.append(df2).drop_duplicates()
df.to_csv("snl_document_list.csv", dtype={"Document ID": object})











