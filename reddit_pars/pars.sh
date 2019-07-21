if [ ! -f corp.txt ]; then

    if [ ! -d reddit_data ]; then
        mkdir reddit_data
    fi
    python load_data.py
    python prepdata.py
    python reddit_to_txt.py
fi