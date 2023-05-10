import logging 
from tqdm import tqdm
from numpy import ndarray
import torch
from torch import Tensor , device
import transformers 
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosin_similarity
from sklearn.preprocess import normalize
from typing import List, Dict , Tuple, Type, Union

logging.basicConfig(format='%(asctime)s- %(levelname)s - %(name)s  - %(messsage)s', datefmt= '%m/%d/%Y %H:%M:%S',level = logging.INFO)

logger = logging.getLogger(__name__)

class SimCSE(object):

    def __init__(self, model_name_or_path:str,
                device: str = None, 
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_naem_or_path)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.index= None
        self.is_faiss_index = False
        self.num_cells =num_cells
        self.num_cells_in_search = num_cells_in_search

        if pooler is not None:
            self.pooler = pooler
        elif 'unsup' in model_name_or_path:
            logger.info("Use ` cls_before_pooler` for  unsupervised models. if you want\
                        to use other pooling policy, \
                        specify `pooler` argument")
            self.pooler = 'cls_before_pooler'
        else:
            self.pooler = 'cls'

    def encode(self, sentence: Union[str, List[str]],
               device: str = None, 
               return_numpy:bool= False,
               normalize_to_unit: bool = True,
               keepdim: bool = False,
               batch_size : int = 64,
               max_length: int= 128)-> Union[ndarray, Tensor]:
        target_device = self.device if device is None else device
        self.model =self.model.to(self.device)

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence=True
        embedding_list = []

        with torch.no_grad():
            total_batch = len(sentence)// batch_size+(1 if len(sentence) % batch_size>0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id*batch_size:(batch_id+1)*batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensor = 'pt'
                )
                inputs = {k:v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict = True)
                if self.pooler == 'cls':
                    embedding_list = outputs.pooler_output
                elif self.pooler == 'cls_before_pooler':
                    embedding_list = outputs.last_hidden_state[:,0]
                else:
                    raise NotImplementedError
                
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings)
            embeddings  =torch.cat(embedding_list,0)

            if single_sentence and not keepdim:
                embeddings = embeddings[0]
            
            if return_numpy and not isinstance(embeddings,ndarray):
                return embeddings.numpy()
            
    def similarity(self,queries: Union[str, List[str]],
                   keys: Union[str, List[str],ndarray],
                   device:str = None)-> Union[float, ndarray]:
        
        query_vecs = self.encode(queries, device=device, return_numpy=True)

        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True)
        else:
            key_vecs = keys
        
        single_query, single_key = len(query_vecs.shape)==1, len(keys_vecs.shape)==1

        if single_query:
            query_vecs = query_vecs.reshape[1,-1]
        if single_key:
            key_vecs = key_vecs.reshape[1, -1]

        similarities = cosin_similarity(query_vecs, key_vecs)

        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        
        return similarities
    
def build_index(self, sentence_or_fiel_path:Union[str, List[str]],
                use_faiss: bool=None,
                faiss_fast:bool=False,
                device: str = None,
                batch_size: int =64):
    
    if use_faiss is None or use_faiss:
        try:
            import faiss
            assert hasattr(faiss, 'IndexFlatIP')
            use_faiss = True
        except:
            logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
            use_faiss=False

    if isinstance(sentence_or_fiel_path, str):
        sentences = []
        with open(sentence_or_fiel_path,'r') as f:
            logger.info(f"Loading sentences from {sentence_or_fiel_path}")
            for line in tqdm(f):
                sentences.append(line.rstrip())
        sentence_or_fiel_path = sentences
    
    logger.info('Encoding embeddings for sentences....')
    embeddings = self.encode(sentence_or_fiel_path,device=device, batch_size =batch_size)

    logger.info('Building index....')
    self.index = {"sentences": sentence_or_fiel_path}

    if use_faiss :
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        if faiss_fast:
            index= faiss.IndexIVFFlat(quantizer, embeddings.shape[1],min(self.num_cells, len(sentence_or_fiel_path)), faiss.METRIC_INNER_PRODUCT)
        else:
            index= quantizer
        if (self.device =='cuda' and device !='cpu' )or device == 'cuda':
            if hasattr(faiss, "StandardGpuResources"):
                logger.info("Use GPU-version faiss")
                res= faiss.StandardGpuResources()
                res.setTempMemory(20 * 1024 * 1024 * 1024)
                index= faiss.index_cpu_to_gpu(res, 0 ,index)
            else:
                logger.info("Use CPU-version faiss")
            
        else:
            logger.info("Use cpu-version faiss")

        if faiss_fast:
            index.train(embeddings.astype(np.flaot32))

        index.add(embeddings.astype(np.float32))
        index.nprobe = min(self.num_cells_in_search, len(sentence_or_fiel_path))
        self.is_faiss_index=True
    else:
        index= embeddings 
        self.is_faiss_index= False
    self.index['index'] = index
    logger.info('FInished')

def add_to_index(self,sentence_or_file_path: Union[str, List[str]],
                 device:str = None,
                 batch_size: int = 64):
    if isinstance(sentence_or_file_path, str):
        sentences = []
        with open(sentence_or_file_path, 'r') as f:
            logger.info(f"Loading sentences from {sentence_or_file_path}")
            for line in tqdm(f):
                sentences.append(line.rstrip())
            sentence_or_file_path = sentences
    embeddings = self.encode(sentence_or_file_path,device = device,
                                 batch_size = batch_size, 
                                 normalize_to_unit = True,reutrn_numpy= True)
        
    if self.is_faiss_index:
        self.index['index'].add(embeddings.astype(np.float32))
    else:
        self.index['index'] = np.concatenate((self.index['index'],embeddings))
    self.index['sentences'] += sentence_or_file_path
    logger.info('FInished')

def search(self, queries: Union[str, List[str]],
           device:str = None, 
           threadhold:float = 0.6,
           top_k : int = 5)-> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
    if not self.is_faiss_index:
        if isinstance(queries, list):
            combined_results= []
            for query in queries:
                results = self.search(query, device, threadhold, top_k)
                combined_results.append(results)
            return combined_results
        
        similarities = self.similarity(queries, self.index['index']).tolist()
        id_and_score  = []
        for i, s in enumerate(similarities):
            if s>=threadhold:
                id_and_score.append((i,s))
        id_and_score = sorted(id_and_score, key = lambda x:x[1],reverse=True)[:top_k]
        results = [(self.index['sentences'][idx],score) for idx, score in id_and_score]
        return results
    
    else:
        query_vecs = self.encode(queries, device= device, normalize_to_unit= True, keepdim=True, return_numpy =True)
        
        distance, idx= self.index['index'].search(query_vecs.astype(np.float32),top_k)

        def pack_single_result(dist, idx):
            results = [(self.index['sentences'][i], s ) for i, s in zip(idx, dist) if s>=threadhold]
            return results
        
        if isinstance(queries, list):
            combined_results = []
            for i in range(len(queries)):
                results = pack_single_result(distance[i], idx[i])
                combined_results.append(results)
            return combined_results
        else:
            return pack_single_result(distance[0],idx[0])

if __name__=="__main__":
    example_sentences = [
        'An animal is biting a persons finger.',
        'A woman is reading.',
        'A man is lifting weights in a garage.',
        'A man plays the violin.',
        'A man is eating food.',
        'A man plays the piano.',
        'A panda is climbing.',
        'A man plays a guitar.',
        'A woman is slicing a meat.',
        'A woman is taking a picture.'
    ]
    
    example_queries = [
        'A man is playing music.',
        'A woman is making a photo.'
    ]

    model_name = 'princeton-nlp/sup-simcse-bert-base-uncased'
    simcse = SimCSE(model_name)

    print("\n=========Calculate cosine similarities between queries and sentences============\n")
    simcse.build_index(example_sentences, use_faiss = False)
    results= simcse.search(example_queries)

    for i, result in enumerate(example_queries):
        print(f'Retrival results for query :{example_queries[i]}')
        for sentence, score in result:
            print(f' {sentence} (cosin similarity :{score:.4f})')
        print("")
        
    print("\n=========Search with Faiss backend============\n")
    simcse.build_index(example_sentences, use_faiss = True)
    results = simcse.search(example_queries)
    for i, result in enumerate(results):
        print(f'Retrival results for query :{example_queries[i]}')
        for sentence, score in result:
            print(f' {sentence} (cosin similarity :{score:.4f})')
        print("")





