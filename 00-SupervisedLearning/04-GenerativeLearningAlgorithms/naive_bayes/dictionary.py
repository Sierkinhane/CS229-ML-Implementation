"""
  author: Sierkinhane
  since: 2018-10-11 15:30:17
  description: create a dictionary
"""
import pandas as pd
from email_content_filter import email_content_filter

def to_dict(path):
    """
    return a dictionary
    """
    with open(path, 'r') as file:
        container = file.readlines()

    # write words to a dictionary "key: value"
    dictionary = dict([(key[:-1], value) for value, key in enumerate(container)])

    return dictionary

def create_a_dict(path):

    email_data = pd.read_csv(path, encoding='utf-8')

    # Create a dictionary
    email_content = email_data['v2']
    email_content_filterd = email_content_filter(email_content)

    dict_list = []
    for item in email_content_filterd:
        dict_list.extend(item)
    
    dictionary = dict([(item, index) for index,item in enumerate(set(dict_list))])

    return dictionary

def combined(dict1, dict2):
    
    list_1 = []
    for k, v in dict1.items():
        list_1.append(k)
    for k, v in dict2.items():
        list_1.append(k)
    print(len(list_1))

    return dict([(item, index) for index,item in enumerate(set(list_1))])


if __name__ == '__main__':
    
    dictionary = to_dict('./words/words.txt')
    own_dict = create_a_dict('./data/spam-utf8.csv')
    p_dict = combined(dictionary, own_dict)
    # print(p_dict)
