import os
base_link = 'files.pushshift.io/reddit/comments/'

def download_comments(tstart, tend, extension='.bz2'):
	month = 12
	while tstart <= tend:
		download_link = base_link + 'RC_{:02d}-{:02d}'.format(tstart[0], tstart[1]) + extension
		os.system('wget ' + download_link)
		tstart[1] += 1
		if tstart[1] > month:
			tstart[1] = 1
			tstart[0] += 1

download_comments([2005, 12], [2017, 11])

# Can not be parsed now
#download_comments((2017, 11), (2018, 10), '.xz')
#download_comments((2018, 11), (2019, 4), '.zst')
