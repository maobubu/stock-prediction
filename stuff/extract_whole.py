#!/usr/bin/python3
'''
extract the news titles from the news
'''
import glob
from datetime import datetime
from tqdm import tqdm
import json
import collections
import re

end_punctuation = '.?!'

raw_news_dir = '/Users/maobu/Documents/datas/'
news_title_dir = '/Users/maobu/Documents/datas/'
#raw_news_folder = ['ReutersNews106521', '20061020_20131126_bloomberg_news']
raw_news_folder = ['ReutersNews106521']

# extract news title from reuters news
reuters_news_file = open(news_title_dir + 'reuters_news_title.txt', 'w')
subfolder_list = glob.glob(raw_news_dir + 'ReutersNews106521/*')
pbar = tqdm(total=len(subfolder_list))
for subfolder in subfolder_list:
    news_date = subfolder.split('/')[-1]
    news_date = datetime.strptime(news_date, '%Y%m%d').strftime('%Y-%m-%d')
    pbar.set_description('Extracting {}'.format(subfolder))
    pbar.update(1)
    for txt_file_dir in glob.glob(subfolder + '/*'):
        with open(txt_file_dir, 'r') as f:
            try:
                str = f.readline()
                if str == '-- \n':
                    str = f.readline()
                news_title = str.split('-- ')[-1]
                news_title = news_title.strip('\n')
                if len(news_title) == 0:
                    continue
                if news_title[-1] not in end_punctuation:
                    news_title += '.\n'
                else:
                    news_title += '\n'
                data = {"date": "", "title": "", "abstract": "", "article": ""}
                j = open('/Users/maobu/Documents/datas/reuters_news.json', 'a+')
                data["date"] = news_date
                data["title"] = news_title.strip('\n')
                #json_o = json.dump(o_data, j)
                #print("\r\n", file=j)
                #j.close()
                #print(data)
                #reuters_news_file.write(news_date + '\t' + news_title)
                line_count = 1
                abstract = ""
                article = ""
                flag = 0
                for line in f.readlines():
                    str_line = line.strip("\n").strip()
                    length = len(str_line)
                    #print(length)
                    if line_count < 7:
                        line_count = line_count + 1
                        continue
                    #if line_count > 8 and line[0] == " " and flag == 0:
                        #flag = 1
                        #article = article + line.strip("\n")
                        #line_count = line_count + 1
                        #continue
                    if line_count >= 7 and flag == 0:
                        abstract = abstract + line.strip("\n")
                        line_count = line_count + 1
                        if length == 0:
                            flag = 1
                            continue
                        if str_line[-1] in end_punctuation:
                            flag = 1
                            continue
                    if line_count >= 7 and flag == 1:
                        article = article + line.strip("\n")
                        line_count = line_count + 1
                data["abstract"] = abstract
                data["article"] = article
                o_data = collections.OrderedDict(data)
                json_o = json.dump(o_data, j)
                print("\r", file=j)
                #print(data)
                j.close()
                #print(data)
            except UnicodeDecodeError as e:
                print('Unicode error')


pbar.close()
reuters_news_file.close()

# extract news title from bloomberg news
#bloomberg_news_file = open(news_title_dir + 'bloomberg_news_title.txt', 'w')
subfolder_list = glob.glob(raw_news_dir + '20061020_20131126_bloomberg_news/*')
pbar = tqdm(total=len(subfolder_list))
for subfolder in subfolder_list:
    news_date = subfolder.split('/')[-1]
    pbar.set_description('Extracting {}'.format(subfolder))
    pbar.update(1)
    for txt_file_dir in glob.glob(subfolder + '/*'):
        #with open(txt_file_dir, 'r+', encoding='gb18030', errors='ignore') as f:
        with open(txt_file_dir, 'r+', errors='ignore') as f:
            str = f.readline()
            if str == '-- \n':
                str = f.readline()
            news_title = str.split('-- ')[-1]
            news_title = news_title.strip('\n')
            if len(news_title) == 0:
                continue
            if news_title[-1] not in end_punctuation:
                news_title += '.\n'
            else:
                news_title += '\n'
            data = {"date": "", "title": ""}
            j = open('/Users/zhangyunke/Desktop/bloomberg_news.json', 'a+')
            data["date"] = news_date
            data["title"] = news_title.strip('\n')
            # json_o = json.dump(o_data, j)
            # print("\r\n", file=j)
            # j.close()
            # print(data)
            # reuters_news_file.write(news_date + '\t' + news_title)
            #line_count = 1
            abstract = ""
            article = ""
            text = ""
            for line in f.readlines():
                text = text + line
                text = text.replace("U.S.  ", "U.S. ").replace("Corp.  ", "Corp. ").replace("Inc.  ", "Inc. ")\
                    .replace("Ltd.  ", "Ltd. ").replace("Co.  ", "Co. ")
            try:
                html_position = re.search('html', text)
                html_position = list(html_position.span())
                cut = int(html_position[1])
                #print(cut)
                text = text[int(cut):]
                case1 = int(text.find('.\n'))
                case2 = int(text.find('. \n'))
                case3 = int(text.find('.  '))
                case4 = int(text.find('  '))
                if case1 != -1:
                    abstract = text[:case1 + 1]
                    article = text[case1 + 1:]
                if case2 != -1:
                    abstract = text[:case2 + 1]
                    article = text[case2 + 1:]
                if case4 != -1:
                    abstract = text[:case4 + 1]
                    article = text[case4 + 1:]
                if case3 != -1:
                    abstract = text[:case3 + 1]
                    article = text[case3 + 1:]
                if abstract == '':
                    data["abstract"] = article.replace("\n", " ").strip()
                    data["article"] = ""
                data["abstract"] = abstract.replace("\n", " ").strip()
                data["article"] = article.replace("\n", " ").strip()
                useless = int(article.find('To contact the'))
                if useless < 10 and useless != -1:
                    data["article"] = ""
                # o_data = collections.OrderedDict(data)
                # json_o = json.dump(o_data, j)
                # print("\r", file=j)
                # print(data)
                # j.close()
                print(data)
            except AttributeError:
                data["abstract"] = abstract.replace("\n", " ").strip()
                data["article"] = article.replace("\n", " ").strip()
                print(data)




