
from itertools import groupby


def split_to_pairs(s):
    new_s = []
    for i in range(1,len(s)):
        new_s.append((s[i-1],s[i]))
    return new_s


def get_reddit_lines(path = 'output/output 1'):
    with open(path, 'r') as f:
        s = f.readlines()
    return s



def split_to_dialogs(all_lines):

    s = [list(group) for k, group in groupby(all_lines, lambda x: x == '\n') if not k]
    return s


def prep_form(s):

    for i in range(len(s)):
        for j in range(len(s[i])):
            s[i][j] = s[i][j].replace('>', '')
            s[i][j] = s[i][j].strip('\n')
            s[i][j] = '[CLS]' + s[i][j] + '[SEP]'
    return s


def final_arr(s):
    new_s = []
    for i in s:
        new_s.extend(split_to_pairs(i))
    return new_s


def to_file(new_s):
    with open('corp.txt','w') as f:
        for (q,a) in new_s:
            f.write(q+'\n'+a+'\n\n')


lines = get_reddit_lines('output/output 1')
#print(lines[:10])
dgs = final_arr(prep_form(split_to_dialogs(lines)))
to_file(dgs)
