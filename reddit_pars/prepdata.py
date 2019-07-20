import bz2
import argparse
import os
import json
import re
import sys

FILE_SUFFIX = ".bz2"
OUTPUT_FILE = "output.bz2"
REPORT_FILE = "RC_report.txt"

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class RedditComment(object):
	def __init__(self, json_object, record_subreddit=False):
		self.body = json_object['body']
		if 'score' in json_object:
			self.score = json_object['score']
		elif 'ups' in json_object and 'down' in json_object:
			self.score = json_object['ups'] - json_object['downs']
		else: raise ValueError("Reddit comment did not include a score attribute. "
			+ "Comment was as follows: " + json_object)
		self.author = json_object['author']
		parent_id = json_object['parent_id']
		if parent_id.startswith('t1_'): self.parent_id = parent_id
		else: self.parent_id = None
		self.child_id = None
		if record_subreddit: self.subreddit = json_object['subreddit']

def parse_main(args):
	if not os.path.isfile(args.config_file):
		print("File not found: {}".format(args.config_file))
		return
	with open(args.config_file, 'r') as f:
		config = json.load(f)
	subreddit_blacklist = set(config['subreddit_blacklist'])
	subreddit_whitelist = set(config['subreddit_whitelist'])
	substring_blacklist = set(config['substring_blacklist'])

	if not os.path.exists(args.input_file):
		print("File not found: {}".format(args.input_file))
		return
	if os.path.isfile(args.logdir):
		print("File already exists at output directory location: {}".format(args.logdir))
		return
	if not os.path.exists(args.logdir):
		os.makedirs(args.logdir)
	subreddit_dict = {}
	comment_dict = {}
	raw_data = raw_data_generator(args.input_file)
	output_handler = OutputHandler(os.path.join(args.logdir, OUTPUT_FILE), args.output_file_size)
	done = False
	total_read = 0
	while not done:
		done, i = read_comments_into_cache(raw_data, comment_dict, args.print_every, args.print_subreddit,
			args.comment_cache_size, subreddit_dict, substring_blacklist, subreddit_whitelist, substring_blacklist)
		total_read += i
		process_comment_cache(comment_dict, args.print_every)
		write_comment_cache(comment_dict, output_handler, args.print_every,
							args.print_subreddit, args.min_conversation_length)
		write_report(os.path.join(args.logdir, REPORT_FILE), subreddit_dict)
		comment_dict.clear()
	print("\nRead all {:,d} lines from {}.".format(total_read, args.input_file))

def read_comments_into_cache(raw_data, comment_dict, print_every, print_subreddit, comment_cache_size,
			subreddit_dict, subreddit_blacklist, subreddit_whitelist, substring_blacklist):
	done = False
	cache_count = 0
	for i, line in enumerate(raw_data):
		if len(line) > 1 and (line[-1] == '}' or line[-2] == '}'):
			comment = json.loads(line)
			if post_qualifies(comment, subreddit_blacklist, # Also preprocesses the post.
				subreddit_whitelist, substring_blacklist):
				sub = comment['subreddit']
				if sub in subreddit_dict:
					subreddit_dict[sub] += 1
				else: subreddit_dict[sub] = 1
				comment_dict[comment['id']] = RedditComment(comment, print_subreddit)
				cache_count += 1
				if cache_count % print_every == 0:
					print("\rCached {:,d} comments".format(cache_count), end='')
					sys.stdout.flush()
				if cache_count > comment_cache_size: break
	else: # raw_data has been exhausted.
		done = True
	print()
	return done, i

def raw_data_generator(path):
	if os.path.isdir(path):
		for walk_root, walk_dir, walk_files in os.walk(path):
			for file_name in walk_files:
				file_path = os.path.join(walk_root, file_name)
				if file_path.endswith(FILE_SUFFIX):
					print("\nReading from {}".format(file_path))
					with bz2.open(file_path, "rt") as raw_data:
						try:
							for line in raw_data: yield line
						except IOError:
							print("IOError from file {}".format(file_path))
							continue
				else: print("Skipping file {} (doesn't end with {})".format(file_path, FILE_SUFFIX))
	elif os.path.isfile(path):
		print("Reading from {}".format(path))
		with bz2.open(path, "rt") as raw_data:
			for line in raw_data: yield line

class OutputHandler():
	def __init__(self, path, output_file_size):
		if path.endswith(FILE_SUFFIX):
			path = path[:-len(FILE_SUFFIX)]
		self.base_path = path
		self.output_file_size = output_file_size
		self.file_reference = None

	def write(self, data):
		if self.file_reference is None:
			self._get_current_path()
		self.file_reference.write(data)
		self.current_file_size += len(data)
		if self.current_file_size >= self.output_file_size:
			self.file_reference.close()
			self.file_reference = None

	def _get_current_path(self):
		i = 1
		while True:
			path = "{} {}{}".format(self.base_path, i, FILE_SUFFIX)
			if not os.path.exists(path): break
			i += 1
		self.current_path = path
		self.current_file_size = 0
		self.file_reference = bz2.open(self.current_path, mode="wt")

def post_qualifies(json_object, subreddit_blacklist,
		subreddit_whitelist, substring_blacklist):
	body = json_object['body']
	post_length = len(body)
	if post_length < 4 or post_length > 200: return False
	subreddit = json_object['subreddit']
	if len(subreddit_whitelist) > 0 and subreddit not in subreddit_whitelist: return False
	if len(subreddit_blacklist) > 0 and subreddit in subreddit_blacklist: return False
	if len(substring_blacklist) > 0:
		for substring in substring_blacklist:
			if body.find(substring) >= 0: return False
	body = re.sub('[ \t\n\r]+', ' ', body)
	body = re.sub('\^', '', body)
	body = re.sub('\\\\', '', body)
	body = re.sub('&lt;', '<', body)
	body = re.sub('&gt;', '>', body)
	body = re.sub('&amp;', '&', body) 
	post_length = len(body)
	if post_length < 4 or post_length > 200: return False
	json_object['body'] = body
	if not json_object['id'].startswith('t1_'): json_object['id'] = 't1_' + json_object['id']
	return True

def process_comment_cache(comment_dict, print_every):
	i = 0
	for my_id, my_comment in comment_dict.items():
		i += 1
		if i % print_every == 0:
			print("\rProcessed {:,d} comments".format(i), end='')
			sys.stdout.flush()
		if my_comment.parent_id is not None: # If we're not a top-level post...
			if my_comment.parent_id in comment_dict: # ...and the parent is in our data set...
				parent = comment_dict[my_comment.parent_id]
				if parent.child_id is None: # If my parent doesn't already have a child, adopt me!
					parent.child_id = my_id
				else: # My parent already has a child.
					parent_previous_child = comment_dict[parent.child_id]
					if parent.parent_id in comment_dict: # If my grandparent is in our data set...
						grandparent = comment_dict[parent.parent_id]
						if my_comment.author == grandparent.author:
							# If I share an author with grandparent, adopt me!
							parent.child_id = my_id
						elif (parent_previous_child.author != grandparent.author
							and my_comment.score > parent_previous_child.score):
							# If the existing child doesn't share an author with grandparent,
							# higher score prevails.
							parent.child_id = my_id
					elif my_comment.score > parent_previous_child.score:
						# If there's no grandparent, the higher-score child prevails.
						parent.child_id = my_id
			else:
				# Parent IDs that aren't in the data set get de-referenced.
				my_comment.parent_id = None
	print()

def write_comment_cache(comment_dict, output_file, print_every,
			record_subreddit=False, min_conversation_length=5):
	i = 0
	prev_print_count = 0
	for k, v in comment_dict.items():
		if v.parent_id is None and v.child_id is not None:
			comment = v
			depth = 0
			if record_subreddit: output_string = "/r/" + comment.subreddit + '\n'
			else: output_string = ""
			while comment is not None:
				depth += 1
				output_string += '> ' + comment.body + '\n'
				if comment.child_id in comment_dict:
					comment = comment_dict[comment.child_id]
				else:
					comment = None
					if depth >= min_conversation_length:
						output_file.write(output_string + '\n')
						i += depth
						if i > prev_print_count + print_every:
							prev_print_count = i
							print("\rWrote {:,d} comments".format(i), end='')
							sys.stdout.flush()
	print()

def write_report(report_file_path, subreddit_dict):
	print("Updating subreddit report file")
	subreddit_list = sorted(subreddit_dict.items(), key=lambda x: -x[1])
	with open(report_file_path, "w") as f:
		for item in subreddit_list:
			f.write("{}: {}\n".format(*item))

import argparse
args = argparse.Namespace()
args.input_file = 'reddit_data/'


if not os.path.exists(args.input_file):
	print(os.path.exists(args.input_file))
	print("File not found: {}".format(args.input_file))
args.logdir = 'output/'
args.config_file = 'parser_config_standard.json'
print(os.path.isfile(args.config_file))
args.comment_cache_size = 1e7
args.output_file_size = 2e8
args.print_every = 1000
args.min_conversation_length = 5
args.print_subreddit = False
parse_main(args)