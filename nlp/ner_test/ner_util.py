import requests
from tqdm.notebook import tqdm
from time import time
import os
from pypinyin import pinyin
import jieba
import re
import logging
import configparser 

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))
config_file = os.path.join(ROOT_DIR, "config.ini")

config = configparser.ConfigParser()
config.read(config_file,encoding='utf8')

slot_url = config['SIT_ENV']['slot_url']
core_dict_file = config['DEFAULT']['core_dict_file']

### 加载pinyin 黑白名单
black_dict = config['BLACK_LIST']
white_dict = config['WHITE_LIST']

cur_path = os.getcwd()
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(format=' %(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

entity = requests.post(slot_url).json()['data']

jieba.re_han_default = re.compile("(.+)", re.U)
jieba.re_userdict = re.compile('^(.+?)(\u0040\u0040[0-9]+)?(\u0040\u0040[a-z]+)?$', re.U)
jieba.re_han_cut_all = re.compile("(.+)", re.U)

def py_withspace(txt, heteronym=True):
    py_result = pinyin(txt.lower(), heteronym=heteronym,style=0)
    return cat_list_py(py_result)

## 数组合并方法
def cat_list_py(list_py):
    result = []
    count = 0
    ss =''
    def func(list_py, count, result, ss):
        cur_list= list_py[count]
        for cur in cur_list:
            if count+1  < len(list_py):
                func(list_py, count+1 , result , ss + " " + cur)
            else:
                result.append(ss + " " + cur)
    func(list_py, count, result, ss)
    return [r.strip() for r in result]

def pinyin_aug_1(txt):
    """
    zh,ch,sh - > z,c,s...
    """
    if 'zh' in txt:
        txt = txt.replace('zh','z')
    if 'ch' in txt:
        txt = txt.replace('ch','c')
    if 'sh' in txt:
        txt = txt.replace('sh','s')
    return txt
    
def pinyin_aug_11(txt):
    if 'ing' in txt:
        txt = txt.replace('ing','in')
    if 'ang' in txt:
        txt = txt.replace('ang','an')
    if 'eng' in txt:
        txt = txt.replace('eng','en')
    if 'ong' in txt:
        txt = txt.replace('ong','un')
    return txt

def pinyin_aug_2(txt):
    """
    z,c,s -> zh,ch,sh ...
    """
    if 'z' in txt and 'zh' not in txt:
        txt = txt.replace('z','zh')
    if 'c' in txt and 'ch' not in txt:
        txt = txt.replace('c','ch')
    if 's' in txt and 'sh' not in txt:
        txt = txt.replace('s','sh')
    return txt

def pinyin_aug_22(txt):
    if 'in' in txt and 'ing' not in txt:
        txt = txt.replace('in','ing')
    if 'an' in txt and 'ang' not in txt:
        txt = txt.replace('an','ang')
    if 'en' in txt and 'eng' not in txt:
        txt = txt.replace('en','eng')
    if 'un' in txt and 'ong' not in txt :
        txt = txt.replace('un','ong')
    return txt

def pinyin_aug_3(txt):
    """
    l<-->n
    """
    if txt.startswith('l'):
        txt = txt.replace('l','n')
    elif txt.startswith('n'):
        txt = txt.replace('n','l')
    return txt

### 获取所有词槽的拼音及变形
re_sub = lambda x:"".join(re.findall('[\u4e00-\u9fa5a-zA-Z0-9]+',x))
def create_han_pin_dict(entity):
    han_pinyin = {}
    han_pinyin_reverse = {}
    for e in tqdm(entity):        
    # {'id': 3010, 'robotId': 10086, 'wordSlotId': 77, 'wordSlotKey': 'KB_MODEL', 'entityName': '节假日', 'createTime': '2023-04-20 11:27:11', 'updateTime': None}
        re_sub_name = re_sub(e['entityName']).lower()
        origin_name = e['entityName']
        entity_id = e['wordSlotId']
        entity_type = e['wordSlotKey']
        py_lists = py_withspace(re_sub_name)
        for py_list in py_lists:
            py_list_size = len(py_list.split())
            if py_list_size>=6 or py_list_size<=1:
                # 长度大于6和小于1的词槽不再进行拼音扩充
                han_pinyin_reverse.setdefault(py_list.lower(),[]).append((origin_name,entity_id,entity_type))
            else:
                list_py = [[] for i in range(py_list_size)]
                for idx, word in enumerate(py_list.split()):
                    list_py[idx].append(pinyin_aug_1(word))
                    list_py[idx].append(pinyin_aug_11(word))
                    list_py[idx].append(pinyin_aug_2(word))
                    list_py[idx].append(pinyin_aug_22(word))
                    list_py[idx].append(pinyin_aug_3(word))
                for pin in set(cat_list_py(list_py)):
                    han_pinyin_reverse.setdefault(pin.lower(),[]).append((origin_name,entity_id,entity_type))
        han_pinyin.setdefault(re_sub_name.lower(),[]).append((origin_name,entity_id,entity_type))
    return han_pinyin,han_pinyin_reverse

try:
    han_pinyin , han_pinyin_reverse = create_han_pin_dict(entity)
except Exception as e:
    raise e

for word in han_pinyin_reverse.keys():
    jieba.add_word(word.lower())
logger.info("用户词槽数据加载成功！！")

if os.path.exists(os.path.join(cur_path,core_dict_file)):    
    with open(os.path.join(cur_path,core_dict_file), 'r' ,encoding='utf8') as f:
        for l in f:
            for word in py_withspace(l.strip(), heteronym=False):
                jieba.add_word(word.lower())
        logger.info("核心文件导入成功！！")
else:
    logger.info(f"未检测到核心文件！！{os.path.join(cur_path,core_dict_file)}")

def black_white_list_filter(txt):
    """
    黑名单：删除误识别出来的结果
    白名单：补充漏识别结果
    """
    #白名单纠错
    final_result = []
    logger.info(f"input query: {txt}")
    for key in white_dict:
        if key in txt:
            txt = txt.replace(key,white_dict[key])
    try:
        pinyin_result = match_pinyin(txt)
    except Exception as e:
        raise e
    txt = re_sub(txt)
    txt_rep = re.sub('[A-Za-z0-9]+','#',txt)

    # 黑名单过滤
    for entity in pinyin_result:
        entityValue = entity['entityValue']
        pos  = entity['index'] 
        ## 如果转拼音后词槽在黑名单中 且 对应黑名单的词在原始问句中 且 位置和转拼音的位置相同 则过滤掉该词槽
        if entityValue in black_dict and black_dict[entityValue] in txt_rep and pos == txt_rep.index(black_dict[entityValue]):
            continue
        else:
            final_result.append(entity)
    return final_result
    
def match_pinyin(txt):
    """
    转拼音模糊匹配
    """
    pinyin_match_result = []   
    txt_py = py_withspace(re_sub(txt),False)[0]
    # for txt_py in txt_pys:
    txt_py_list = txt_py.split()
    cut_words = jieba.lcut(txt_py)
    logger.info(f"拼音分词结果: {cut_words}")
    seg_result = []
    
    for idx, word in enumerate(cut_words):
        seg_result.append((idx,word))
    pos = 0
    for idx, word in enumerate(seg_result):
        p, w = word
        if w==' ':
            continue
        logger.info(f'*********w:::{w}******')
        if w not in txt_py_list and len(set(w.split()) & set(txt_py_list))==0:
            continue
        if w in han_pinyin_reverse:
            entity_infos = han_pinyin_reverse[w]
            logger.info(f'*********entity_infos:::{entity_infos}******')
            entity_name, entity_id, entity_type = entity_infos[0]
            # 拼音得分默认0.8
            tmp_entity = {"entityName": entity_type, "entityId":entity_id, "entityValue": entity_name, "score": 0.8, "from": "py_slot", 'index': pos}
            pinyin_match_result.append(tmp_entity)
        if ' ' not in w:
            pos += 1
        else:
            pos += len(w.split())
    return pinyin_match_result

if __name__=='__main__':
    txt ='与332甲基442氨基联苯发生反应的试剂有哪些，与1,2-2甲基肼发生反应的试剂有哪些'
    #氢氧化啦,氧气r120皇城,哪些实验室空调没有打开？氨水，西安,213分析以71'
    start = time()
    for r in (black_white_list_filter(txt)):
        print(r)
    print(f"*********耗时：{1000*(time()-start)} ms************\n")