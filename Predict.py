# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
import sys
import torch
import json
import random
import argparse
import collections
import torch.nn as nn
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import  BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from brain import KnowledgeGraph
from multiprocessing import Process, Pool
#from BM25 import similarity
from transformers import AdamW
import numpy as np
from torch.utils.data import DataLoader
import jieba
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score



class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


def add_knowledge_worker(params):

    p_id, sentences, columns, kg, vocab, args = params
    
    sentences_num = len(sentences)
    dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 10000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')
        
        try:
            if len(line) == 2:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]]
                
                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
        
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

                dataset.append((token_ids, label, mask, pos, vm))
            
            elif len(line) == 3:
                label = int(line[columns["label"]])
                text = CLS_TOKEN + line[columns["text_a"]] + SEP_TOKEN + line[columns["text_b"]] + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1

                dataset.append((token_ids, label, mask, pos, vm))
            
            elif len(line) == 4:  # for dbqa
                qid=int(line[columns["qid"]])
                label = int(line[columns["label"]])
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text = CLS_TOKEN + text_a + SEP_TOKEN + text_b + SEP_TOKEN

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool")

                token_ids = [vocab.get(t) for t in tokens]
                mask = []
                seg_tag = 1
                for t in tokens:
                    if t == PAD_TOKEN:
                        mask.append(0)
                    else:
                        mask.append(seg_tag)
                    if t == SEP_TOKEN:
                        seg_tag += 1
                
                dataset.append((token_ids, label, mask, pos, vm, qid))
            else:
                pass
            
        except:
            print("Error line: ", line)
    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")
    # parser.add_argument("--BM25_path", type=str, required=True,
    #                     help="Path of the BM25.")
    # parser.add_argument("--BM25Type_path", type=str, required=True,
    #                     help="Path of the BM25Type_path.")
    # parser.add_argument("--BM25_test_path", type=str, required=True,
    #                     help="Path of the BM25Test_path.")
    # parser.add_argument("--Stopword_path", type=str, required=True,
    #                     help="Path of the Stopword.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=False, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass

    args.labels_num = len(labels_set) 
    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build classification model.
    model = BertClassifier(args, model)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    
    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vms_batch = vms[i*batch_size: (i+1)*batch_size]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vms_batch = vms[instances_num//batch_size*batch_size:]

            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files, predicate=True)

    def read_dataset(path, workers_num=1):

        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                sentences.append(line)
        sentence_num = len(sentences)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append((i, sentences[i*sentence_per_block: (i+1)*sentence_per_block], columns, kg, vocab, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params)

        return dataset

    # Evaluation function.
    def evaluate(args, is_test, metrics='f1'):
        
        if is_test:
            dataset = read_dataset(args.test_path, workers_num=args.workers_num)
           
        else:
            dataset = read_dataset(args.dev_path, workers_num=args.workers_num)
        
       
        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([example[3] for example in dataset])
        vms = [example[4] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()
       # BM25Rank,train_type_dict=BM25()

        # start=0
        # end=batch_size
        
        #if not args.mean_reciprocal_rank:
            
        if not args.mean_reciprocal_rank:
            print("計算F1")
            ls=[]
            data=[]
            ls_gold=[]
            ls_pred=[]
          
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):
               
                
                # RankResult=[]
                # BatachBM25=BM25Rank[start:end]
                # start=start+batch_size
                # end=start+batch_size
                # vms_batch = vms_batch.long()
                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)
                ls.append(label_ids_batch)
                
                with torch.no_grad():
                    try:
                        loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                    except:
                        print(input_ids_batch)
                        print(input_ids_batch.size())
                        print(vms_batch)
                        print(vms_batch.size())

                logits = nn.Softmax(dim=1)(logits)
                pred = torch.argmax(logits, dim=1)
               
                gold = label_ids_batch
                
                # for (l,B) in zip(logits,BatachBM25):
              
                    # sort_label=torch.sort(l,descending=True)
                    # sort_dict=dict()
                    # for (score,label) in zip(sort_label[0],sort_label[1]):
                        # label=int(label.cpu().numpy())
                        # score=float(score.cpu().numpy())
                        # sort_dict[label]=score
                    # print(sorted(B, key = lambda x : x[1]))
                    # print(max(sort_dict.values()))
                    # maximum=max(sort_dict.values())
                    # keys = [key for key, value in sort_dict.items() if value == maximum] 
                    # print(keys)
                    # sys.exit()
                    # ReRank= dict()
                    # TotalScore=0
                    # for b in B:
                        # TotalScore+=b[1]
                    
                    # for b in B:
                        # if(sort_dict[train_type_dict[b[0]]]>0.953):
                        #     ReRank[b[0]]=sort_dict[train_type_dict[b[0]]]  #b[1]+
                        # else:b[1]+sort_dict[train_type_dict[b[0]]]((b[1]/949.11)*100)+
                        # ReRank[b[0]]=sort_dict[train_type_dict[b[0]]]
        
                    # maximum = max(ReRank.values()) 
                    # keys = [key for key, value in ReRank.items() if value == maximum]
                    # result=train_type_dict[keys[0]]
                    # gold = label_ids_batch
                    # RankResult.append(result)
   
                # pred=torch.Tensor(RankResult).cuda()
                print("Pred")
                print(pred)
                # gold = label_ids_batch
                print("Gold")
                print(gold)
                for (p,g) in zip(pred.cpu().numpy(),gold.cpu().numpy()):
                    ls_gold.append(g)
                    ls_pred.append(int(p))
              
                # for j in range(pred.size()[0]):
                #     confusion[pred[j], gold[j]] += 1
                # correct += torch.sum(pred == gold).item()
            print(ls_gold)
            print(ls_pred)
            print("準確率")
            print(accuracy_score(ls_gold, ls_pred))
        
            f1 = f1_score(ls_gold,ls_pred, average='macro')
            p = precision_score(ls_gold, ls_pred, average='macro')
            r = recall_score(ls_gold, ls_pred, average='macro')
            print("F1 Score:")
            print(f1)
            print("Precision:")
            print(p)
            print("Recall:")
            print(r)
         
        else:
            print("計算MRR")
            rank_pos=[]
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, vms_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vms)):

                vms_batch = torch.LongTensor(vms_batch)

                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                vms_batch = vms_batch.to(device)

                with torch.no_grad():
                    loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
                logits = nn.Softmax(dim=1)(logits)
                
                for l,g in zip(logits,label_ids_batch):
                   
                    sort_label=torch.sort(l,descending=True)
                    for (idx,p) in enumerate(sort_label[1]):
                        if(p==g):
                            rank_pos.append(idx)
            MRR=0
            for score in rank_pos:
                MRR+=1/(score+1)
            print("MRR:",MRR/1035)
           
    
    def BM25_readfile():
        #讀取BM25的訓練資料(標準問句斷過詞)
        BM25_train_data=[]
        with open(args.BM25_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
            
                BM25_train_data.append(line.replace("\r", "").replace("\n", ""))
        #讀取stop word
        stopWords=[]
        with open(args.Stopword_path, 'r', encoding='UTF-8') as file:
            for data in file.readlines():
                data = data.strip()
                stopWords.append(data)
        return stopWords,BM25_train_data
    
    def BM25():
        stopWords,BM25_train_data=BM25_readfile()  
        count_l1_sim=0
        count_num_l1=0
        count_l0_sim=0
        count_num_l0=0
        correct=0
        predict=[]
        similarity_ls=[]
        for sentence in BM25_train_data[1:]:
            S1=sentence.strip().split('\t')[0]
            S2=sentence.strip().split('\t')[1]
            words1 = [w for w in S1.split() if w not in stopWords]
            S1=" ".join(words1)
            words2 = [w for w in S2.split() if w not in stopWords]
            S2=" ".join(words2)

            sim=similarity.ssim(S1,S2,model='bm25')
            similarity_ls.append(sim)
            if(sim<11):
                predict_label=0
                predict.append(predict_label)

            if(sim>11):
                predict_label=1
                predict.append(predict_label)
      
     
    # Evaluation phase.
    print("Final evaluation on the test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))
    accuracy=evaluate(args, True,metrics='Acc')
  
    print("準確率:")
    print(accuracy)

if __name__ == "__main__":
    main()