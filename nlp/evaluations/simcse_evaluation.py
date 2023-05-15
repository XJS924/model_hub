import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


logging.basicConfig(format="%(sactime)s: %(message)s", level = logging.DEBUG)

PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'


sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names,scores):
    tb = PrettyTable()
    tb.field_names= task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        help='Transformers model name or path')
    parser.add_argument('--pooler',type=str,choices=['cls', 'cls_before_pooler','avg', 'avg_top2',
                                                     'avg_first_last'],
                        default= 'cls',help='which pooler to use')
    parser.add_argument('--mode', type = str,
                        choices=['dev','test', 'fasttest'],default='test',
                        help='what evaluation mode to use')
    
    parser.add_argument('--task_set', type=str,choices=['sts', 'transfer', 'full', 'na',],
                        default='sts',help='what set of tasks to evaluate on ')
    
    parser.add_argument('--tasks',type=str, nargs="+",
                        default=['STS12','STS13','STS14','STS15','STS16','MR','CR',"MPQA",])
    
    



