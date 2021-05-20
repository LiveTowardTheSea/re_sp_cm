class reformer_config:
    def __init__(self):
        self.name='reformer'
        self.layer_num = 4
        self.round = 6 
        self.k_dim = 64
        self.head = 4
        self.d_model = 256
        self.attend = 1
        self.p_drop = 0.1
        self.bucket = 16
        self.d_ff = 1024
        self.regularization = 0.008
        self.lr = 0.001
        self.lr_decay = 0.05
        self.epoch_num = 30
        self.momentum = 0.9

class sparse_config:
    def __init__(self):
        self.name='sparse'
        self.layer_num = 4
        self.head = 4
        self.d_model = 256
        self.k_dim = 64
        self.stride = 64
        self.local_head = 2
        self.c = 16
        self.p_drop = 0.1
        self.d_ff = 1024
        self.regularization = 0.008
        self.lr = 0.001
        self.lr_decay = 0.05
        self.epoch_num = 30
        self.momentum = 0.9
    
class compress_config:
    def __init__(self):
        self.name='compress'
        self.layer_num = 4
        self.layer = 'LMLM'
        self.d_model = 256
        self.head = 4
        self.k_dim = 64
        self.block_size = 64
        self.compression = 8
        self.p_drop = 0.1
        self.d_ff = 1024
        self.regularization = 0.008
        self.lr = 0.001
        self.lr_decay = 0.05
        self.epoch_num = 30
        self.momentum = 0.9