import torch
import load_data
from models import hi_xlnet
import time
import math
import os
from torch.optim.swa_utils import SWALR
from args import parser
from tqdm import tqdm
import torch.nn as nn
import warnings
import optuna
from sklearn import metrics
import os


warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

class TrainModel():

    def save_model(self, step, accuracy1):
        if not os.path.isdir('saved_models'):
            os.mkdir("saved_models")

        model_path = os.path.join("saved_models",
                                  "epoch_{}_{}_seed-{}_bs-{}_lr-{}_step-{}_acc-{}_type-{}_p=5.pair".format(self.epochs,self.desc, self.seed,
                                                                                              self.batch_size,
                                                                                              self.learning_rate, step,
                                                                                              accuracy1,
                                                                                              self.model_size))

        print('model has saved')
        torch.save(self.xlnet_model.state_dict(), model_path)

    def account_implicite(self,t_outputs_all,t_targets_all,t_implicit):

        out_pre = torch.argmax(t_outputs_all, -1)
        im_num, ex_num = 0, 0
        im_cor, ex_cor = 0, 0
        for index in range(len(t_implicit)):
            all_num = len(t_implicit)
            # 'False': 0,
            # 'True': 1,
            if t_implicit[index] == 0:
                ex_num = ex_num + 1
                if t_targets_all[index] == out_pre[index]:
                    ex_cor = ex_cor + 1
            else:
                im_num = im_num + 1
                if t_targets_all[index] == out_pre[index]:
                    im_cor = im_cor + 1

        acc_im = im_cor / im_num
        acc_ex = ex_cor / ex_num
        print('account results: all : {}, im_num: {}, ex_num: {}, acc im : {}, acc ex {}'.format(all_num,im_num,ex_num,acc_im,acc_ex))

    def account_aspect(self,t_outputs_all,t_targets_all,t_aspect):
        out_pre = torch.argmax(t_outputs_all, -1)
        has_num, no_num = 0, 0
        has_cor, no_cor = 0, 0
        all_num = len(t_aspect)
        for index in range(len(t_aspect)):
            # 'aspect==null': 0,
            # 'aspect is not null': 1,
            if t_aspect[index] == 0:
                no_num = no_num + 1
                if t_targets_all[index] == out_pre[index]:
                    no_cor = no_cor + 1
            else:
                has_num = has_num + 1
                if t_targets_all[index] == out_pre[index]:
                    has_cor = has_cor + 1

        acc_has = has_cor / has_num
        acc_no = no_cor / no_num
        print('account aspect results: all : {}, has_num: {}, no_num: {}, acc has : {}, acc no {}'.format(all_num, has_num, no_num, acc_has, acc_no))

    def _reset_params(self):
        for p in self.xlnet_model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.model_size = args.model_size
        self.learning_rate = args.lr_start
        self.anneal_to = args.lr_end
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.negs = args.num_negs
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        if args.test_file:
            self.test_file = args.test_file
        else:
            self.test_file = args.dev_file
        self.margin = args.margin
        self.desc = args.model_description
        self.seed = args.seed
        self.datatype = args.data_type
        self.max_len = args.max_len
        self.bestacc = 0
        self.bestacc_sen = 0
        self.train_data = load_data.LoadConnData(self.train_file, self.batch_size, self.model_size, self.device,
                                                 self.datatype, self.max_len)
        self.dev_data = load_data.LoadConnData(self.test_file,self.batch_size, self.model_size, self.device,
                                               self.datatype, self.max_len) #

        self.xlnet_model = hi_xlnet.Hi_XL_GCN(args, self.negs, self.batch_size, self.margin, self.device)
        self.criterion = nn.CrossEntropyLoss()

        self.xlnet_model = self.xlnet_model.to(self.device)

        self.initializer = args.initializer
        no_decay = ['bias', 'LayerNorm.weight']
        # self._reset_params()
        # n是每层的名称，p是每层的参数
        optimizer_grouped_parameters = [
            # 分层学习率（以transformers中的BertModel为例）
            {'params': self.xlnet_model.sen_xl.parameters(), 'weight_decay': 0.001, 'lr': 2e-5},
            {'params': self.xlnet_model.tf1.parameters(), 'weight_decay':  0.00001, 'lr': args.lr_start},
        ]


        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)  # , lr=self.learning_rate
        self.scheduler = SWALR(self.optimizer, anneal_strategy="linear", anneal_epochs=args.lr_anneal_epochs,
                               swa_lr=args.lr_end)
        self.total_loss = 0.0
        self.total_sen_loss = 0.0
        self.total_doc_loss = 0.0
        self.total_cl_loss = 0.0
        self.eval_interval = args.eval_interval

    def train_xlnet_model(self):
        train_loader = self.train_data.data_loader()

        self.epochs = 20

        start = time.time()
        self.xlnet_model.train()

        for epoch in range(self.epochs):
            n_correct, n_total = 0, 0
            print("Epoch: {}/{}".format(epoch + 1, self.epochs))
            for step, data in enumerate(tqdm(train_loader)):
                try:
                    pos_input, neg_input, sentence = data
                except Error as e:
                    print(e)
                    continue

                sentence_inputs = [sentence['sen_index'][0, :].to(self.device),
                                   sentence['catagory'][0, :].to(self.device),
                                   sentence['text_indices'][0, :].to(self.device),
                                   sentence['input_ids'][0, :].to(self.device),
                                   sentence['attention_mask'][0, :].to(self.device),
                                   sentence['token_starts'][0, :].to(self.device),
                                   sentence['token_start_mask'][0, :].to(self.device),
                                   sentence['left_indices'][0, :].to(self.device),
                                   sentence['aspect_indices'][0, :].to(self.device),
                                   sentence['catagory_indices'][0, :].to(self.device),
                                   sentence['dependency_graph'][0, :].to(self.device),
                                   sentence['aspect_in_text'][0, :].to(self.device),
                                   sentence['aspect_in_text_mask'][0, :].to(self.device),
                                   ]

                sentence_targets = sentence['polarity'][0, :].to(self.device)
                sentence_implicit = sentence['implicit'][0, :].to(self.device)
                # if len(sentence_implicit)==1:
                #     continue

                # document -level
                pos_input = [pos_input[0]['pos_input_ids'].to(self.device),
                             pos_input[0]['pos_attention_mask'].to(self.device)]
                    # pos_input[0].to(self.device)

                neg_inputs = []
                for i in range(len(neg_input)):
                    neg_inputs.append([neg_input[i]['neg_input_ids'].to(self.device),neg_input[i]['neg_attention_mask'].to(self.device)])

                pos_score, neg_scores, sentence_outputs = self.xlnet_model([pos_input, neg_inputs, sentence_inputs])

                sen_loss = self.criterion(sentence_outputs, sentence_targets)

                docu_loss =0.5*self.xlnet_model.contrastiveLoss(pos_score, neg_scores)

                loss = sen_loss + docu_loss #+ cl_loss  # sa_loss + 0.1*

                loss.backward()

                torch.nn.utils.clip_grad_norm_(parameters=self.xlnet_model.parameters(), max_norm=2.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # bert_optimizer.step()
                self.total_sen_loss += sen_loss.item()
                self.total_doc_loss += docu_loss.item()

                self.total_loss += loss.item()

                n_correct += (torch.argmax(sentence_outputs, -1) == sentence_targets).sum().item()

                n_total += len(sentence_outputs)
                train_acc = n_correct / n_total

                if step % self.eval_interval == 0 and step > 0:
                    bestacc,_ =self.eval_model(step, start)
                    self.scheduler.step()
                    print("Steps: {} total Loss: {} sen loss: {} cl loss: {}, doc loss: {},train sen Acc: {}".format(step, self.total_loss,
                                                                                                             self.total_sen_loss,
                                                                                                             self.total_cl_loss,
                                                                                                             self.total_doc_loss,train_acc))
                    self.total_loss = 0.0
                    self.total_sen_loss = 0.0
                    self.total_doc_loss = 0.0
                    self.total_cl_loss = 0.0

        return self.bestacc_sen

    def eval_model(self, step, start):

        dev_loader = self.dev_data.data_loader()
        self.xlnet_model.eval()
        correct = 0.0
        total = 0.0
        t_targets_all, t_outputs_all, t_implicit, t_aspect = None, None, None, None
        self.y_pres = []
        self.y_tures = []
        sen_n_test_correct, sen_n_test_total = 0, 0

        with torch.no_grad():

            for data in dev_loader:
                try:
                    pos_input, neg_input, sentence = data
                except Error as e:
                    print(e)
                    continue

                # sentence -level training
                sentence_targets = sentence['polarity'][0, :].to(self.device)
                sen_implicit = sentence['implicit'][0, :].to(self.device)
                sen_aspect_or_not = sentence['aspect_or_not'][0, :].to(self.device)

                sentence_inputs = [sentence['sen_index'][0, :].to(self.device),
                                   sentence['catagory'][0, :].to(self.device),
                                   sentence['text_indices'][0, :].to(self.device),
                                   sentence['input_ids'][0, :].to(self.device),
                                   sentence['attention_mask'][0, :].to(self.device),
                                   sentence['token_starts'][0, :].to(self.device),
                                   sentence['token_start_mask'][0, :].to(self.device),
                                   sentence['left_indices'][0, :].to(self.device),
                                   sentence['aspect_indices'][0, :].to(self.device),
                                   sentence['catagory_indices'][0, :].to(self.device),
                                   sentence['dependency_graph'][0, :].to(self.device),
                                   sentence['aspect_in_text'][0, :].to(self.device),
                                   sentence['aspect_in_text_mask'][0, :].to(self.device),
                                   ]
                pos_input = [pos_input[0]['pos_input_ids'].to(self.device), pos_input[0]['pos_attention_mask'].to(self.device)]
                neg_inputs = []
                for i in range(len(neg_input)):
                    neg_inputs.append([neg_input[i]['neg_input_ids'].to(self.device),
                                       neg_input[i]['neg_attention_mask'].to(self.device)])

                pos_score, neg_scores, sentence_outputs = self.xlnet_model([pos_input, neg_inputs, sentence_inputs])
                sen_n_test_correct += (torch.argmax(sentence_outputs, -1) == sentence_targets).sum().item()
                sen_n_test_total += len(sentence_outputs)

                max_neg_score = torch.max(neg_scores, -1).values
                if pos_score > max_neg_score:
                    correct += 1.0
                total += 1.0

                if t_targets_all is None:
                    t_targets_all = sentence_targets
                    t_outputs_all = sentence_outputs
                    t_implicit = sen_implicit
                    t_aspect = sen_aspect_or_not
                else:
                    t_targets_all = torch.cat((t_targets_all, sentence_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, sentence_outputs), dim=0)
                    t_implicit = torch.cat((t_implicit,sen_implicit), dim=0)
                    t_aspect = torch.cat((t_aspect,sen_aspect_or_not), dim=0)


        self.xlnet_model.train()

        # end = time.time()
        acc = correct / total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        sen_test_acc = sen_n_test_correct / sen_n_test_total

        print('Model: ', self.desc, 'Seed: ', self.seed)
        print("DEV EVAL Steps: {} sen Acc: {} F1 {} doc acc: {}".format(step, sen_test_acc, f1, acc))  #
        if step > 0:
            if sen_test_acc > self.bestacc_sen and sen_test_acc > 0.925:
                self.account_implicite(t_outputs_all, t_targets_all, t_implicit)
                self.account_aspect(t_outputs_all, t_targets_all, t_aspect)
                self.bestacc_sen = sen_test_acc
                self.desc = 'sen'
                self.save_model(step, sen_test_acc)

        # return self.y_tures,self.y_pres
        return self.bestacc_sen, acc

def main(trial):
    opt = parser.parse_args()
    # # optuna setting for tuning hyperparameters
    # opt.alpha = trial.suggest_uniform('alpha', 0.6, 0.9)
    # opt.alpha = trial.suggest_int('d_model', 0.1, 0.9, 0.05)
    # opt.lr_start = trial.suggest_uniform('lr_start', 0.000005, 0.00001)
    # opt.lr_end = trial.suggest_uniform('lr_end', 0.00001, 0.00005)
    # opt.dropout = trial.suggest_uniform('dropout_rate', 0.1, 0.9)
    initializers = {
        # xavier_uniform_
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    opt.initializer = initializers[opt.initializer]

    print('[Info] parameters: {}'.format(opt))

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    start = time.time()
    Trainer = TrainModel(opt)
    bestacc_sen = Trainer.train_xlnet_model()
    with open('result.txt', 'a') as f:
        f.write('[Info] parameters: {} \n'.format(opt))
        f.write(str(bestacc_sen)+"\n")
        f.close()
    return bestacc_sen

if __name__ == '__main__':

    study = optuna.create_study(direction="maximize")
    study.optimize(main, n_trials=3)

    df = study.trials_dataframe()

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))