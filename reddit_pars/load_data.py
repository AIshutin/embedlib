from bs4 import BeautifulSoup
import requests
import os
url = 'http://files.pushshift.io/reddit/comments/'
r = requests.get(url)
soup = BeautifulSoup(r.content)
print(r)
list_of_files = soup.find_all('tr',{'class' : "file"})
list_of_files = list(map(lambda x: x.a['href'],list_of_files))
filtr = lambda x: x[-4 :] == '.bz2'
list_of_files = list(filter(filtr, list_of_files))
import urllib.request
print('Beginning file download ...')
k = 0
for i in list_of_files:
    print('Downloaded ', k, ' files\nContinue? [yes/no]')
    flag = input()
    if flag == 'no':
    break
    dwn = url  + i.strip('./')
    dwn.replace(' ','_')
    if os.path.exists('./reddit_data/'+i):
    print('File already exist, download again? [yes/no]')
    flag = input()
    if flag == 'no':
    continue
    urllib.request.urlretrieve(dwn, './reddit_data/'+i)
    k += 1

print('Downloaded files to /reddit_data/')
