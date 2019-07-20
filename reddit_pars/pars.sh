if [ ! -f corp.txt ]; then
    mkdir reddit_data
    python load_data.py
    python prepdata.py
    python reddit_to_txt.py
fi