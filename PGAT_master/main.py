from classes.NERController import NERController


if __name__ == '__main__':
    MyController=NERController(
        charWindowSize=3,
        maxSentenceLength=250,
        lr=0.0005,
        decay_rate=0.03,
        epoch=50,
        batchSize=1,
        char_emb_file='gigaword_chn.all.a2b.uni.ite50.vec',  # choose from [gigaword_chn.all.a2b.uni.ite50.vec,  bert]
        word_emb_file='ctb.50d.vec',  # choose from [ctb.50d.vec, sgns.merge.word]
        hidden_dim=200,
        trust_rank=0.8,
        pgat_head=5, #  max(d_char,d_word)%pgat_head==0
        seq_head=5,   #d_w %seq_head==0
        att_head=10, #   hidden_dim%att_head==0
        dropout=0.2,
        att_dropout=0.2,
        seq_feature_model='relative_transformer',#choose from [transformer relative_transformer]
        context_feature_model='default',#choose from [lstm or default]
        use_gpu=True,
        use_bert=True,
        optimizer='Adam',   #choose from [Adam, Adamax, SGD]
        dropout_trainable=True,
        source_type='weibo_all', #dataset name, choose from [resume   weibo_all  ontonote  ecommerce]
        which_step_to_print=300,
        cuda_version=0,
        seq_encoding_model='lstm',#sequence encoding layer, choose from [lstm  cnn  transformer relative_transformer]
        layer_num=4,
        random_seed=1023
    )
    MyController.train()
    # MyController.test()

