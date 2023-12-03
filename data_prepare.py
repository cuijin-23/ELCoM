import xml.etree.ElementTree as ET
import pickle
import numpy as np
import os
import random
import itertools
import spacy as sp
import re
from tqdm import tqdm

def get_permutations(sentence_count, permutation_count):
    res = []
    if sentence_count < 3:
        total_count = 0
        original_order = [x for x in range(sentence_count)]
        perms = set(itertools.permutations(original_order))
        for perm_order in perms:
            perm_order = list(perm_order)
            if np.all(np.array(perm_order)==np.array(original_order)):
                continue
            if total_count >= permutation_count:
                break
            total_count+=1
            res.append(perm_order)
    else:
        original_order = np.array([x for x in range(sentence_count)])
        prev_perms = []
        for j in range(permutation_count):
            perm_order = np.random.permutation(sentence_count)
            perm_str = ','.join([str(z) for z in perm_order])
            while np.all(perm_order==original_order) or perm_str in prev_perms:
                perm_order = np.random.permutation(sentence_count)
                perm_str = ','.join([str(z) for z in perm_order])
            prev_perms.append(perm_str)
            res.append(perm_order)
    random.shuffle(res)
    return res

def store_results(train_path, dev_path, res):
    with open(train_path, 'wb') as dfile:
        pickle.dump(res, dfile)


def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text

def store_results_test(test_path,res):
    with open(test_path, 'wb') as dfile:
        pickle.dump(res, dfile)

# base_dir = os.path.dirname(os.path.realpath(__file__))

def load_file(file_path):
    sentences = []
    with open(file_path, 'rb') as dfile:
        for line in dfile.readlines():
            sentences.append(line.strip())
    return sentences

def process_sentences(reviews_list, max_sequence_length):
    sents = reviews_list
    sents = [x for x in sents if x != '<para_break>' and x != '']

    if max_sequence_length == -1:
        return sents, False

    processed_sents = []
    word_length = 0
    is_truncated = False
    for x in sents:
        words = [w.text for w in sp(x)]
        if word_length + len(words) > max_sequence_length:
            is_truncated = True
            break
        word_length += len(words)
        processed_sents.append(x)
    return processed_sents, is_truncated

def process_semeval_2016():
    # load the raw data
    # prepare the coherence dataset
    train_filepath = './dataset/train_pair=5_absa_Implicit_Labeled.pkl'
    dev_filepath = './dataset/dev.pkl'
    test_filepath = './dataset/test_pair=5_absa_Implicit_Labeled.pkl'
    polar = {
        'negative': -1,
        'neutral': 0,
        'positive': 1,
    }
    implicit = {
        'False': 0,
        'True': 1,
    }

    train_filename = './dataset/ABSA16_Restaurants_Train_SB1_v2_Implicit_Labeled.xml'
    test_filename = './dataset/EN_REST_SB1_TEST_Implicit_Labeled.xml.gold'#
    reviews = ET.parse(test_filename).getroot().findall('Review')

    res = []
    permutations = 4
    one_sent_doc_count = 0
    for r in tqdm(reviews):
        sentences = r.find('sentences').getchildren()
        sentences_list = []

        for sentence in sentences:
            aspect_sen = []
            ori_sen = sentence.find('text').text
            # ori_sen = clean_text(ori_sen)
            ori_sen = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", ori_sen)
            aspect_sen.append(ori_sen)
            if (sentence.find('Opinions') == None) or (sentence.find('Opinions').find('Opinion') == None) :
                sentences_list.append(aspect_sen)
                continue
            Opinions = sentence.find('Opinions').getchildren()
            #针对多个aspect的问题：
            # len_op = 0
            # last_target = Opinions[0].attrib.get('target')
            for opin in Opinions:
                # len_op +=1
                target = opin.attrib.get('target')
                catagory = opin.attrib.get('category').lower()
                catagory = catagory.replace('#', " ")
                print(target)
                # if len_op>1 and last_target == target:
                #     continue
                replace_sen = ori_sen.replace(target, "$T$")
                aspect_sen.append(replace_sen)
                aspect_sen.append(target)
                aspect_sen.append(catagory)
                aspect_sen.append(polar[opin.attrib.get('polarity')])
                aspect_sen.append(implicit[opin.attrib.get('implicit_sentiment')])
            sentences_list.append(aspect_sen)

        sents_list, is_truncated = process_sentences(sentences_list, -1)

        permutation_ordering = get_permutations(len(sentences_list), permutations)

        print('the ordering of sentence', permutation_ordering)
        print(sentences_list)

        temp = {}
        temp['pos'] = sents_list

        neg_per = {}
        for idx, perm_order in enumerate(permutation_ordering):
            neg_list = []
            for i_ in perm_order:
                neg_list.append(sentences_list[i_][0])
            neg_per[idx] = neg_list
            # else: temp['neg'].append(tt)
        temp['negs'] = neg_per
        res.append(temp)

    # random.shuffle(res)
    # store_results(train_filepath,dev_filepath, res)
    store_results_test(test_filepath,res)
    print('the result has saved!')
    return res


def main():
    corpora = process_semeval_2016()
    print(corpora)

if __name__ == '__main__':
    main()
