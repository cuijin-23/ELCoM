import pickle
import torch
from transformers import XLNetTokenizer, XLNetModel
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import spacy
import re

nlp_st = StanfordCoreNLP('./stanford-corenlp-4.5.2')

def track_tokens(tokens, max_len, tokenizer):
    """Segment each token into subwords while keeping track of
    token boundaries.
    Parameters
    ----------
    tokens: A list of strings, representing input tokens.
    Returns
    ----------
    A tuple consisting of:
        - token_start_mask:
        An array with size (max_len) in which word starts tokens is 1 and all other subwords is 0.
        - token_start:
        An array of indices into the list of subwords, indicating
        that the corresponding subword is the start of a new
        token. For example, [1, 3, 4, 7] means that the subwords
        1, 3, 4, 7 are token starts, while all other subwords
        (0, 2, 5, 6, 8...) are in or at the end of tokens.
        This list allows selecting Bert hidden states that
        represent tokens, which is necessary in sequence
        labeling.
    """
    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])

    token_start = torch.zeros(max_len, dtype=torch.long)
    token_start[0:len(token_start_idxs)] = torch.tensor(token_start_idxs)

    token_start_mask = torch.zeros(max_len, dtype=torch.long)
    token_start_mask[token_start_idxs] = 1

    return token_start, token_start_mask

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    # 补充或者截断
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


def dependency_adj_matrix_stand(text):
    # document = nlp_st(text)
    dep_outputs = nlp_st.dependency_parse(text)
    seq_len = len(dep_outputs)
    tokens = nlp_st.word_tokenize(text)

    # '''查找根结点对应的索引'''
    root_index = []
    for i in range(len(dep_outputs)):
        if dep_outputs[i][0] == 'ROOT':
            root_index.append(i)

    # '''修改依存关系三元组'''
    new_dep_outputs = []
    for i in range(len(dep_outputs)):
        for index in root_index:
            if i + 1 > index:
                tag = index

        if dep_outputs[i][0] == 'ROOT':
            dep_output = (dep_outputs[i][0], dep_outputs[i][1], dep_outputs[i][2] + tag)
        else:
            dep_output = (dep_outputs[i][0], dep_outputs[i][1] + tag, dep_outputs[i][2] + tag)
        new_dep_outputs.append(dep_output)

    head_list = []
    for i in range(len(tokens)):
        for dep_output in new_dep_outputs:
            if dep_output[-1] == i + 1:
                head_list.append(int(dep_output[1]))

    matrix = np.zeros((seq_len+1, seq_len+1), dtype=np.float32)
    for i in range(len(head_list)):
        j = head_list[i]
        if j != 0:
            matrix[i, j - 1] = 1
            matrix[j - 1, i] = 1

    return matrix


# nlp = spacy.load('en_core_web_sm')

# def dependency_adj_matrix(text):
#     # https://spacy.io/docs/usage/processing-text
#     document = nlp(text)
#     seq_len = len(text.split())
#     matrix = np.zeros((seq_len, seq_len)).astype('float32')
#
#     for token in document:
#         if token.i < seq_len:
#             matrix[token.i][token.i] = 1
#             # https://spacy.io/docs/api/token
#             for child in token.children:
#                 if child.i < seq_len:
#                     matrix[token.i][child.i] = 1
#
#     return matrix

class ConnDataset(Dataset):
    def __init__(self, fname, model, device, datatype, max_len):

        self.fname = fname
        self.device = device
        self.data = pickle.load(open(fname, 'rb'))
        # self.sen_tokenizer = Tokenizer4Bert('bert-{}-uncased'.format(model))
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-{}-cased'.format(model))
        self.truncount = 0
        self.datatype = datatype
        self.max_len = max_len

    def sort_and_pad(self, data):
        sorted_data = data
        batches = self.pad_data(sorted_data)
        return batches

    def pad_data(self, batch_data):

        batch_attention_mask = []
        batch_input_ids = []
        batch_catagory=[]
        batch_dependency_graph = []
        batch_text_indices = []
        batch_polarity = []
        batch_implicit = []
        batch_left_indices = []
        batch_catagory_indices = []
        batch_aspect_indices = []
        batch_sen_index = []
        batch_token_start=[]
        batch_token_start_mask = []

        batch_aspect_in_text=[]
        batch_aspect_in_text_mask=[]
        batch_aspect_or_not = []

        if len(batch_data) == 0:
            return

        for item in batch_data:
            sen_index,catagory, text_indices, input_ids, attention_mask,token_starts, token_start_mask, left_indices, aspect_indices, catagory_indices, dependency_graph, aspect_in_text, aspect_in_text_mask, aspect_or_not, polarity, implicit = item['sen_index'], item['catagory_index'], item['text_indices'], item['input_ids'],\
                                                                                                                                                 item['attention_mask'],item['token_starts'], item['token_start_mask'], item['left_indices'], \
                                                                                                                                                 item['aspect_indices'], item['catagory_indices'], item['dependency_graph'], \
                                                                                                                                                 item[ 'aspect_in_text'], item['aspect_in_text_mask'], item['aspect_or_not'], item['polarity'], item['implicit']

            # xlnet for sa
            batch_sen_index.append(sen_index)
            batch_catagory.append(catagory)
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

            batch_text_indices.append(text_indices)
            batch_left_indices.append(left_indices)

            batch_token_start.append(token_starts)
            batch_token_start_mask.append(token_start_mask)
            batch_aspect_in_text.append(aspect_in_text)
            batch_aspect_in_text_mask.append(aspect_in_text_mask)
            batch_aspect_indices.append(aspect_indices)
            batch_catagory_indices.append(catagory_indices)
            batch_aspect_or_not.append(aspect_or_not)
            batch_polarity.append(polarity)
            batch_implicit.append(implicit)
            batch_dependency_graph.append(dependency_graph)

            # batch_polarity_mask.append(polarity_mask)

        catagory_group = []
        csd = {}
        for i in range(len(batch_catagory)):
            cs = batch_catagory[i]
            if cs not in csd:
                csd[cs] = [i]
            else:
                csd[cs].append(i)
        for i in range(len(batch_catagory)):
            cata_sen = csd[batch_catagory[i]]
            cata_sen = cata_sen + [-1] * (len(batch_catagory) - len(cata_sen))
            catagory_group.append(cata_sen)

        return { \
            'sen_index': torch.tensor(batch_sen_index),
            'catagory': torch.tensor(catagory_group,dtype=torch.long),
            'text_indices': torch.tensor(batch_text_indices),
            'input_ids': torch.tensor(batch_input_ids),
            'attention_mask': torch.tensor(batch_attention_mask),
            'token_starts': torch.tensor(batch_token_start),
            'token_start_mask': torch.tensor(batch_token_start_mask),
            'left_indices': torch.tensor(batch_left_indices, dtype=torch.long),
            'aspect_indices': torch.tensor(batch_aspect_indices,dtype=torch.long),
            'catagory_indices':torch.tensor(batch_catagory_indices,dtype=torch.long),
            'dependency_graph': torch.tensor(batch_dependency_graph),
            'aspect_in_text': torch.tensor(batch_aspect_in_text, dtype=torch.long),
            'aspect_in_text_mask': torch.tensor(batch_aspect_in_text_mask),
            'aspect_or_not': torch.tensor(batch_aspect_or_not),
            'polarity': torch.tensor(batch_polarity), \
            'implicit': torch.tensor(batch_implicit),
            # 'polarity_mask': torch.tensor(batch_polarity_mask)
        }

    # 'bert4gcn': ['input_ids', 'attention_mask', 'token_type_ids', 'dependency_graph', 'token_starts',
    #              'token_start_mask', 'text_raw_indices', 'aspect_in_text', 'aspect_in_text_mask']

    def pad_ids(self, ids, maxlen):
        if len(ids) < maxlen:
            padding_size = maxlen - len(ids)
            padding = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token) for i in range(padding_size)]
            ids = ids + padding
        else:
            print('longer than 600', len(ids))
            ids = ids[:maxlen]
            self.truncount += 1

        return ids

    def prepareData(self, idx):

        # print('idx:',idx)
        if len(self.data[idx]['negs']) == 0:
            idx =idx + 1
        if len(self.data[idx]['negs']) == 0:
            idx =idx + 1

        pos_sen_list = []
        pos_doc = self.data[idx]['pos']
        max_length = 150

        for sen in pos_doc:
            pos_sen_list.append(sen)

        all_sentecne = []
        pos_input = []

        for i in range(len(pos_sen_list)):

            if len(pos_sen_list[i]) == 1:  # or pos_sen_list[i][2] == 'NULL'
                continue
            for j in range(0, len(pos_sen_list[i]) - 1, 5):
                text_left, _, text_right = [s.lower().strip() for s in pos_sen_list[i][j + 1].partition("$T$")]
                aspect = pos_sen_list[i][j + 2]
                catagory = pos_sen_list[i][j + 3]
                # cata_group = {}
                # cata_group['catagory'] = catagory
                # num = num+1
                # target-based
                text = pos_sen_list[i][0]
                # print(text)

                original_line = text + ' <sep> ' + catagory + ' <sep> ' + '<cls>'

                if aspect == 'NULL':
                    # original_line = text_left + ' <sep> '+'<cls>'
                    text = text_left
                else:
                    # original_line = text_left + " " + aspect + " " + text_right + ' <sep> ' + aspect + ' <sep> '+'<cls>'
                    text = text_left + " " + aspect + " " + text_right


                # dependency_adj
                adj_matrix = dependency_adj_matrix_stand(text)
                dependency_graph = np.pad(adj_matrix, \
                                          ((0, max_length - adj_matrix.shape[0]),
                                           (0,  max_length - adj_matrix.shape[0])), 'constant')

                encodings = self.tokenizer.encode_plus(original_line, add_special_tokens=False,
                                                  return_tensors='pt', return_token_type_ids=False,
                                                  return_attention_mask=True, pad_to_max_length=True)
                # return_offsets_mapping = True



                # input_ids = pad_sequences(encodings['input_ids'], maxlen=150, dtype=torch.Tensor, truncating="post",
                #                           padding="post")
                input_ids = encodings['input_ids'][0]
                input_ids = pad_and_truncate(input_ids, max_length)

                input_ids = input_ids.astype(dtype='int64')

                attention_mask = encodings['attention_mask'][0]
                attention_mask = pad_and_truncate(attention_mask, max_length)
                attention_mask = attention_mask.astype(dtype='int64')

                catagory_indics = self.tokenizer.tokenize(catagory)
                catagory_indics = self.tokenizer.convert_tokens_to_ids(catagory_indics)
                left_indices = self.tokenizer.tokenize(text_left)
                left_indices = self.tokenizer.convert_tokens_to_ids(left_indices)

                catagory_ins = catagory_indics + [0] * (max_length - len(catagory_indics))
                catagory_ins = np.array(catagory_ins)

                if max_length < len(left_indices):
                    # print('longer left indices: ', len(left_indices))
                    left_indices = left_indices[:max_length]
                    left_indices = np.array(left_indices)
                else:
                    left_indices = left_indices + [0] * (max_length - len(left_indices))
                    left_indices = np.array(left_indices)


                text_indices = self.tokenizer.tokenize(text)
                text_indices = self.tokenizer.convert_tokens_to_ids(text_indices)
                if max_length < len(text_indices):
                    # print('longer text_indices indices: ', len(text_indices))
                    text_indices = text_indices[:max_length]
                    text_indices = np.array(text_indices)
                else:
                    text_indices = text_indices + [0] * (max_length - len(text_indices))
                    text_indices = np.array(text_indices)

                token_start, token_start_mask = track_tokens(text.split(), max_length, self.tokenizer)
                token_start = token_start.detach().numpy()
                token_start_mask = token_start_mask.detach().numpy()
                if aspect == 'NULL':
                    aspect_indices = [0] * (max_length)
                    aspect_indices = np.array(aspect_indices)
                    left_context_len = np.sum(left_indices != 0)
                    aspect_len = np.sum(aspect_indices != 0)
                    aspect_in_text = [left_context_len.item(), (left_context_len + aspect_len - 1).item()]
                    aspect_in_text_mask = aspect_indices
                    aspect_or_not = 0
                else:
                    aspect_indices = self.tokenizer.tokenize(aspect)
                    aspect_indices = self.tokenizer.convert_tokens_to_ids(aspect_indices)
                    aspect_indices = aspect_indices + [0] * (max_length - len(aspect_indices))
                    aspect_indices = np.array(aspect_indices)
                    left_context_len = np.sum(left_indices != 0)
                    aspect_len = np.sum(aspect_indices != 0)
                    aspect_in_text = [left_context_len.item(), (left_context_len + aspect_len - 1).item()]
                    aspect_in_text_mask = torch.zeros(max_length, dtype=torch.long)
                    aspect_in_text_mask[left_context_len.item(): (left_context_len + aspect_len).item()] = 1
                    aspect_in_text_mask = aspect_in_text_mask.detach().numpy()
                    aspect_or_not = 1
                    # aspect_in_text_mask = aspect_in_text_mask.astype(dtype='int64')

                polarity = pos_sen_list[i][j + 4]
                polarity = int(polarity) + 1
                aspect_or_not = int(aspect_or_not)
                implicit = int(pos_sen_list[i][j + 5])

                sent_data = {
                    'sen_index': i,
                    'catagory_index': catagory,
                    'text_indices': text_indices,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_starts': token_start,
                    'token_start_mask': token_start_mask,
                    'left_indices': left_indices,
                    'aspect_indices': aspect_indices,
                    'catagory_indices': catagory_ins,
                    'dependency_graph': dependency_graph,
                    'aspect_in_text': aspect_in_text,
                    'aspect_in_text_mask': aspect_in_text_mask,
                    'aspect_or_not': aspect_or_not,
                    'polarity': polarity,
                    'implicit': implicit
                }
                all_sentecne.append(sent_data)

        sentenc_batches = self.sort_and_pad(all_sentecne)

        neg_docs = []
        if self.datatype == 'single':
            neg_docs = [self.data[idx]['neg']]
        elif self.datatype == 'multiple':
            num = len(self.data[idx]['negs'])
            for l in range(num):
                neg_docs.append(self.data[idx]['negs'][l])

        pos_span = []
        for pos_list in pos_doc:
            pos_span.append(pos_list[0].lower().strip())

        pos_span = '<sep> '.join(pos_span)
        pos_span = pos_span + ' <sep> '+'<cls>'

        encodings = self.tokenizer.encode_plus(pos_span, add_special_tokens=False,
                                               return_tensors='pt', return_token_type_ids=None,
                                               return_attention_mask=True, pad_to_max_length=True)

        # pos_input_ids = pad_sequences(encodings['input_ids'], maxlen=self.max_len, dtype=torch.Tensor, truncating="post", padding="post")
        # pos_input_ids = pos_input_ids.astype(dtype='int64').flatten()

        pos_input_ids = encodings['input_ids'][0]
        pos_input_ids = pad_and_truncate(pos_input_ids, self.max_len)
        pos_input_ids = pos_input_ids.astype(dtype='int64')

        pos_attenion_mask = encodings['attention_mask'][0]
        pos_attenion_mask = pad_and_truncate(pos_attenion_mask, self.max_len)
        pos_attenion_mask = pos_attenion_mask.astype(dtype='int64')


        pos_data = {
            'pos_input_ids': torch.tensor(pos_input_ids),
            'pos_attention_mask': torch.tensor(pos_attenion_mask),
        }
        pos_input.append(pos_data)

        neg_input = []
        for neg_doc in neg_docs:
            neg_span = []
            for neg_list in neg_doc:
                neg_span.append(neg_list.lower().strip())

            neg_span = ' <sep> '.join(neg_span)
            neg_span = neg_span + ' <sep> ' + '<cls>'
            encodings = self.tokenizer.encode_plus(neg_span, add_special_tokens=False,
                                                   return_tensors='pt', return_token_type_ids=None,
                                                   return_attention_mask=True, pad_to_max_length=True)
            # neg_input_ids = pad_sequences(encodings['input_ids'], maxlen=self.max_len, dtype=torch.Tensor,
            #                               truncating="post", padding="post")

            neg_input_ids = encodings['input_ids'][0]
            neg_input_ids = pad_and_truncate(neg_input_ids, self.max_len)
            neg_input_ids = neg_input_ids.astype(dtype='int64')



            neg_attenion_mask = encodings['attention_mask'][0]
            neg_attenion_mask = pad_and_truncate(neg_attenion_mask, self.max_len)
            neg_attenion_mask = neg_attenion_mask.astype(dtype='int64')


            neg_data = {
            'neg_input_ids': torch.tensor(neg_input_ids),
            'neg_attention_mask': torch.tensor(neg_attenion_mask),
            }
            neg_input.append(neg_data)

        # pos_input = self.tokenizer.build_inputs_with_special_tokens(pos_ids)

        return pos_input, neg_input, sentenc_batches

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx)
        index = self.prepareData(idx, )
        if index[2] == None:
            return self.prepareData(idx + 1)
        return index


class LoadConnData():
    def __init__(self, fname, batch_size, model, device, datatype, max_len):
        self.fname = fname
        self.batch_size = batch_size
        self.dataset = ConnDataset(fname, model, device, datatype, max_len)
        print('ss')

    def data_loader(self):
        dataSampler = SequentialSampler(self.dataset)
        loader = DataLoader(dataset=self.dataset, sampler=dataSampler, batch_size=self.batch_size)
        return loader