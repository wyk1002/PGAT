# PGAT:
This is the implementation of our arxiv paper “PGAT: Polymorphic Graph Attention Network for Chinese NER”, which is a Polymorphic Graph Attention Network that aims to capture the dynamic relationship between characters and words from multiple dimensions to enhance the representation of characters.

## Requirement:
1.  python 3.8.3
2.  torch 1.7.1+cu110
3.  transformers 4.2.1 
## Input format:
CoNLL format, with each character and its label split by a whitespace in a line. The "BMESO" or "BIESO" tag scheme is prefered.
```python
成 B-GPE
都 E-GPE
电 B-ORG
信 E-ORG
智 O
慧 O
生 O
活 O
```

## Pretrain embedding:
The pretrained embeddings(word embedding, char embedding) are the same with [Lattice LSTM](https://aclanthology.org/P18-1144/)
## Main file:

path | illustrates
---- | -----
/cache/emb_dic/ |  the dictionaries constructed by  char and word embedding
/cache/gaz_embedding/ | the pretrained embedding, you can replace it 
 /cache/inf/ | the trie constructed by word embedding
/cache/input_data/ | the processed data for each dataset 
/cache/variable/| the saved model. /base/ is base model, and /bert/ is the PGAT+BERT model
/classes/ |  the implement of PGAT
/resource/ | the dataset 
 /result/ | the output files that can help you tune parameters 
## Run the code:
step 1, download pretrained embeddings:  
1)character embeddings(gigaword_chn.all.a2b.uni.ite50.vec) from [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu pan](https://pan.baidu.com/s/1pLO6T9D).  
2)word embeddings (ctb.50d.vec) from [Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) or [Baidu pan](https://pan.baidu.com/s/1pLO6T9D).  
        then, put them in the */cache/gaz_embedding/* folder. 
        
step 2, name your datasets with "train.bmes, dev.bmes, test.bmes" and put them in the */resource/*  folder.

step 3, build Controller as follows:
```python
from classes.NERController import NERController
import tensorflow as tf
    MyController=NERController(
        charWindowSize=3,
        maxSentenceLength=200,
        lr=0.0005,
        decay_rate=0.05,
        epoch=30,
        batchSize=1,
        emb_dim=50,
        hidden_dim=512,
        data_type=tf.float32,
        att_head=5,
        dropout=0.2,
        att_dropout=0.2,
        use_bert=True,
        dropout_trainable=True,
        source_type='weibo_all', #dataset name, choose from [resume   weibo_all  ontonote  ecommerce]
        which_step_to_print=500,
        cuda_version=1,
        model_name='lstm',#sequence encoding layer, choose from [lstm  cnn  transformer]
        random_seed=857
    )
```
	
step 4, execute *MyController.train()* or *MyController.test()* to train or test the model.
## Run your own data:
step 1: put your data set on the path：*/resource/*, and name your file like we do.
 
step 2: find the init_ctg() mathod in  */classes/NERController.py*, add your data information in the mathod.

step 3: modify the parameters in Controller your built according to your dataset. Then, run the code on your own data.

## Cite:
@article{Wang_2022,	doi = {10.1016/j.eswa.2022.117467},	
url = {https://doi.org/10.1016/j.eswa.2022.117467 },	year = 2022,	month = {may},	publisher = {Elsevier {BV}},	
pages = {117467},	author = {Yuke Wang and Ling Lu and Yang Wu and Yinong Chen},	
title = {Polymorphic Graph Attention Network for Chinese {NER}},	
journal = {Expert Systems with Applications}}
