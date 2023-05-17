import SentenceEvaluator
import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score
import numpy as np
from typing import List
from ..readers.InputExample import InputExample

logger = logging.getLogger(__name__)

class BinaryClassificationEvaluator(SentenceEvaluator):

    def __init__(self, sentence1: List[str], sentence2: List[str], labels:List[int], name : str= '', batch_size:int= 32, 
                 show_progress_bar: bool= False, 
                 write_csv: bool = True):
        self.sentence1= sentence1
        self.sentence2 = sentence2
        self.labels = labels

        assert len(self.sentence1) == len(self.sentence2)
        assert len(self.sentence1)==len(self.labels)

        for label in labels:
            assert (label ==0 or label ==1)

        self.write_csv = write_csv

        self.name = name
        self.batch_size= batch_size
        
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar= show_progress_bar

        self.csv_file = "binary_classification_evaluation" + ("_" + name if name else '') + "_results.csv"

        self.csv_heads =  ["epoch", "steps",
                            "cossim_accuracy", "cossim_accuracy_threshold", "cossim_f1", "cossim_precision", "cossim_recall", "cossim_f1_threshold", "cossim_ap",
                            "manhattan_accuracy", "manhattan_accuracy_threshold", "manhattan_f1", "manhattan_precision", "manhattan_recall", "manhattan_f1_threshold", "manhattan_ap",
                            "euclidean_accuracy", "euclidean_accuracy_threshold", "euclidean_f1", "euclidean_precision", "euclidean_recall", "euclidean_f1_threshold", "euclidean_ap",
                            "dot_accuracy", "dot_accuracy_threshold", "dot_f1", "dot_precision", "dot_recall", "dot_f1_threshold", "dot_ap"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1= []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        
        return cls(sentences1, sentences2, scores, **kwargs)
    
    def __call__(self, model, output_path: str= None, epoch : int = -1, steps: int =-1)-> float:

        if epoch!=-1:
            if steps ==-1:
                out_txt = f"after epoch {epoch}:"
            else:
                out_txt = f"in epoch {epoch} after {steps} steps"
        else:
            out_txt = ':'

        logger.info(f"Binary Accuracy Evaluation of the model on {self.name} dataset {out_txt}")

        scores= self.compute_metrics(model)

        main_score = max(scores[short_name]['ap'] for short_name in scores)
        
        file_output_data = [epoch, steps]

        for header_name in self.csv_heads:
            if '_' in header_name:
                sim_fct , metric = header_name.split('_', maxsplit = 1)
        
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, new_line = '', mode= 'w', encoding='utf-8') as f:
                    writer = csv.write(f)
                    writer.writerow(self.csv_heads)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline='', model='a', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)
        return main_score
    
    def compute_metrics(self, model):
        sentences = list(self.sentence1 + self.sentence2)
        embeddings = model.encode(sentences, batch_size= self.batch_size, show_progress_bar = self.show_progress_bar,
                                  convert_to_numpy = True)
        emb_dict = {sent: emb for sent , emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentence1]
        embeddings2 = [emb_dict[sent] for sent in self.sentence2]

        cosine_socres= 1- paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        embeddings1_np = np.asanyarray(embeddings1)
        embeddings2_np = np.asanyarray(embeddings2)

        dot_scores= [np.dot(embeddings1_np[i], embeddings2_np[i]) for i in range(len(embeddings1_np))]

        labels = np.asanyarray(self.labels)

        output_scores= {}
        for short_name, name ,scores , reverse in [['cossim', 'Cosine_Similarity', cosine_socres, True], ['manhanttan', 'Manhattan_Distances', manhattan_distances, False],
                                                    ['euclidean', 'Euclidean-Distance', euclidean_distances, False], ['dot', 'Dot-Product', dot_scores, True]]:
            acc, acc_threshold, = self.find_best_acc_and_threshold(scores, labels, reverse)
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
            ap = average_precision_score(labels, scores, *( 1 if reverse else -1))

            logger.info("Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(name, acc * 100, acc_threshold))
            logger.info("F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(name, f1 * 100, f1_threshold))
            logger.info("Precision with {}:          {:.2f}".format(name, precision * 100))
            logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
            logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))

            output_scores[short_name] = {
                "accuracy":acc,
                'accuracy_threshold': acc_threshold,
                "f1":f1, 
                'f1_threshold': f1_threshold,
                'precision':precision,
                'recall':recall,
                'ap':ap
            }

        return output_scores
    
    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))
        rows = sorted(rows, key =lambda x:x[0], reverse=high_score_more_similar)

        max_acc= 0
        best_threshold = -1

        remaining_negatives = sum(labels == 0)
        
        for i in range(len(rows) -1):
            score ,label =rows[i]
            if label == 1:
                positive_so_far +=1
            else:
                remaining_negatives -=1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i+1][0]) /2

        return max_acc, best_threshold
    
    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows= list(zip(scores, labels))

        rows = sorted(rows, key = lambda x:x[0] , reverse = high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0

        nextract = 0
        ncorrect = 0

        total_num_duplicates = sum(labels)

        for i in range(len(rows)-1):
            score , label = rows[i]
            nextract += 1
            if label == 1:
                ncorrect+=1

            if ncorrect> 0:
                precision = ncorrect /nextract
                recall = ncorrect / total_num_duplicates
                f1 =2 * precision * recall /( precision + recall)
                if f1 > best_f1:
                    best_f1 = f1 
                    best_precision = precision 
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i+1][0])/2
        return best_f1, best_precision, best_recall, threshold










