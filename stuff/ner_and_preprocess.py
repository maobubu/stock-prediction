import tqdm
import json
import re
reuters_list = open('/Users/zhangyunke/Desktop/data/NER_reuters.txt')
bloomberg_list = open('/Users/zhangyunke/Desktop/data/NER_bloomberg.txt')

reuters_in = open('/Users/zhangyunke/Desktop/data/reuters_news.json', 'r')
bloomberg_in = open('/Users/zhangyunke/Desktop/data/bloomberg_news.json', 'r')
reuters_output = open('/Users/zhangyunke/Desktop/data/reuters_title_ner.json', 'w')
bloomberg_output = open('/Users/zhangyunke/Desktop/data/bloomberg_title_ner.json', 'w')
sp_500_file = open('/Users/zhangyunke/Desktop/data/sp500_code.txt')

sp_500_list = []
for sp in sp_500_file.readlines():
    sp_500_list.append(sp.split("\n"))

entity_list = []
news_date_list = []
news_number_list = []
filter_date_list = []

def cleanSentences(string):  # delete all the punctuation in data
    string = string.lower().replace("<br />", " ")
    return re.sub(re.compile("[^A-Za-z0-9 ]+"), "", string.lower())

def cleanparent(string):
    string = re.sub(r"\(.*?\)", " ", string)
    return string

def cleannumber(string):
    nonum = ""
    word_list = list(string.split())
    for m in range(len(word_list)):
        try:
            word_list[m] = int(word_list[m])
            word_list[m] = "xxx"
            nonum = nonum + " " + word_list[m]
        except ValueError:
            nonum = nonum + " " + word_list[m]
            continue
    return nonum


def clean_bervi(string):
    return string.replace("'ll", " will").replace("'re", " are").replace("'ve", " have").replace("'d", " would")\
        .replace("n't", " not").replace("'s", " is").replace("'m", " am")


def clean_meaningless(string):
    return string.replace(" a ", " ").replace(" an ", " ").replace(" the ", " ").replace(" am ", " ")\
        .replace(" is ", " ").replace(" be ", " ").replace(" are ", " ").replace(" was ", " ").replace(" were ", " ")


def clean_news(raw):
    raw_title = raw.lower().replace("%", "percent").replace('&', 'xiahuaxian')  # get raw title
    raw_title = raw_title.replace("'s", "").replace("u.s.", "america").replace("american", "america")
    raw_title = raw_title.replace("update 1-", "").replace("update 2-", "").replace("update 3-", "").replace(
        "update 4-", "") \
        .replace("update 5-", "").replace("update 6-", "").replace("update 7-", "").replace("update 8-", "") \
        .replace("update 9-", "").replace("update 10-", "")
    raw_title = raw_title.replace("wall st.", "wall street").replace("s&p 500", "standardpoor").replace("s&p",
                                                                                                        "standardpoor")
    raw_title = raw_title.replace("factbox:", "").replace("analysis:", "").replace("insight:", "").replace("advisory:",
                                                                                                           "") \
        .replace("bernanke:", "").strip(" ")
    raw_title = clean_bervi(raw_title)
    no_pun_title = cleanSentences(raw_title)
    no_pun_title = no_pun_title.replace('corrected', "").replace('UPDATE\s\d-', "").replace("'s", "").replace('factbox', "")\
                    .replace('instant view', "").replace('snap analysis', '').replace('exclusive', '').replace('timeline', '')\
                    .replace('highlights', '').replace('correction', '').replace('scenarios', '').replace('analysis', '')
    no_pun_title = clean_meaningless(no_pun_title).replace('xiahuaxian', '_')
    no_num_title = cleannumber(no_pun_title).strip().replace("xxx xxx", "xxx").replace("xxx xxx xxx", "xxx").replace('wrapup xxx', "")
    return no_num_title


def make_real_list(file):
    for i in tqdm.tqdm(file.readlines()):
        i = i.split()
        #print(i[-1])
        if (len(i)) <= 2:
            continue
        if i[-1] == 'DATE':
            continue
        elif i[-1] == 'NUMBER':
            continue
        elif i[-1] == 'MONEY':
            continue
        elif i[-1] == 'DURATION':
            continue
        elif i[-1] == 'PERCENT':
            continue
        elif i[-1] == 'TIME':
            continue
        elif i[-1] == 'URL':
            continue
        del i[-1]
        entity_list.append(" ".join(i).lower())

def connect_entity(entity_list, title):
    for j in range(len(entity_list)):
        if entity_list[j] in title:
            x = "_".join(entity_list[j].split())
            title = title.replace(entity_list[j], x)
    return title


def count_this_news(date, date_list, number_list):
    if date in date_list:
        flag = date_list.index(date)
        number_list[flag] += 1
    else:
        date_list.append(date)
        number_list.append(1)


def check_this_news(article):
    for code in range(len(sp_500_list)):
        if sp_500_list[code] in article:
            return True
    return False
    # Return if this news should be kept

make_real_list(reuters_list)
make_real_list(bloomberg_list)
entity_list = list(set(entity_list))
title = connect_entity(entity_list, "clinton yellen, owner prince, jess blackburn, republican democratic, gary gerhardt")
print(title)

# Find out which days has too much news
for reuters in tqdm.tqdm(reuters_in.readlines()):
    dic = json.loads(reuters)
    count_this_news(dic["date"], news_date_list, news_number_list)
for bloomberg in tqdm.tqdm(bloomberg_in.readlines()):
    dic = json.loads(bloomberg)
    count_this_news(dic["date"], news_date_list, news_number_list)
print(len(news_date_list))
print(len(news_number_list))
for i in range(len(news_date_list)):
    if news_number_list[i] > 80:
        filter_date_list.append(news_date_list[i])
print(len(filter_date_list))
news_counter = 0
data = {"date": "", "title": "", "abstract": ""}
# Then filter it, do preprocess and NER
reuters_in.close()
bloomberg_in.close()

reuters_input = open('/Users/zhangyunke/Desktop/data/reuters_news.json', 'r')
bloomberg_input = open('/Users/zhangyunke/Desktop/data/bloomberg_news.json', 'r')

for reuters in tqdm.tqdm(reuters_input.readlines()):
    dic = json.loads(reuters)
    if dic["date"] in filter_date_list:
        flag = check_this_news(dic["article"])
        if flag is True:
            data["date"] = dic["date"]
            dic["title"] = clean_news(dic["title"])
            data["title"] = connect_entity(entity_list, dic["title"])
            dic["abstract"] = clean_news(dic["abstract"])
            data["abstract"] = connect_entity(entity_list, dic["abstract"])
            json_o = json.dump(data, reuters_output)
            print("\r", file=reuters_output)
            news_counter += 1
        else:
            continue
    else:
        data["date"] = dic["date"]
        dic["title"] = clean_news(dic["title"])
        data["title"] = connect_entity(entity_list, dic["title"])
        dic["abstract"] = clean_news(dic["abstract"])
        data["abstract"] = connect_entity(entity_list, dic["abstract"])
        json_o = json.dump(data, reuters_output)
        print("\r", file=reuters_output)
        news_counter += 1

for bloomberg in tqdm.tqdm(bloomberg_input.readlines()):
    dic = json.loads(bloomberg)
    if dic["date"] in filter_date_list:
        flag = check_this_news(dic["article"])
        if flag is True:
            data["date"] = dic["date"]
            dic["title"] = clean_news(dic["title"])
            data["title"] = connect_entity(entity_list, dic["title"])
            dic["abstract"] = clean_news(dic["abstract"])
            data["abstract"] = connect_entity(entity_list, dic["abstract"])
            json_o = json.dump(data, bloomberg_output)
            print("\r", file=bloomberg_output)
            news_counter += 1
        else:
            continue
    else:
        data["date"] = dic["date"]
        dic["title"] = clean_news(dic["title"])
        data["title"] = connect_entity(entity_list, dic["title"])
        dic["abstract"] = clean_news(dic["abstract"])
        data["abstract"] = connect_entity(entity_list, dic["abstract"])
        json_o = json.dump(data, bloomberg_output)
        print("\r", file=bloomberg_output)
        news_counter += 1
