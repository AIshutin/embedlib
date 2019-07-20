if [ ! -f corp.txt ]; then
	mkdir reddit_data && cd reddit_data \
					&& python3 ../download_reddit_dataset.py \
					&& cd ..				 
	python3 reddit_parse.py
	bzip2 -d output/*.bz2
	rm output/*.bz2
	cat output/output* > output/reddit_comments.txt
	python3 reddit_to_bert.py
fi