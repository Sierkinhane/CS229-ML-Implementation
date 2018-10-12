"""
  author: Sierkinhane
  since: 2018-10-10 13:53:34
  description: spam-filter 
"""
import numpy as np 
import pandas as pd
import re

def email_content_filter(email_content):
    """
    sign filtering
    '. ~ ! @ # $ % ^ & * ( ) - _ = + [ ] { } " " : ; \ | < > / ?'
    1. 检测到有符号的字符串就从该符号分开, 并删除该符号
    """

    # according space to split a sentence into a list of words
    list_1 =[ temp.split(' ') for _, temp in enumerate(email_content)]

    list_2 = []
    for index, content in enumerate(list_1):
        # for one email one list to store in list_2
        temp_list = []
        for _ in content:
            # lower the words
            _ = _.lower()
            # clear some signs
            temp = re.split(r'[\..~!@#$%^&*(,)-_=+[\]{}"":;|<>/?\\]', _)
            temp_list.append(temp[0])

        list_2.append(temp_list)

    return list_2

if __name__ == '__main__':
    
    # read a csv data
    email_data = pd.read_csv('./data/spam-utf8.csv', encoding='utf-8')

    # Transform into two categories 0 and 1
    ins = {'ham': '0', 'spam': '1'}
    email_data = email_data.replace({'v1': ins})
    email_data['v1'] = email_data['v1'].astype('int32') # change into type of integer
    # print(email_data['v1'])

    # Create a dictionary
    email_content = email_data['v2']
    email_content_filterd = email_content_filter(email_content)
    print(email_content_filterd)

