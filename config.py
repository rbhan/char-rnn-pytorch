class Config(object):
    def __init__(self,
                 cuda,
                 data_dir,
                 save_dir,
                 rnn_size,
                 num_layers,
                 # rnn_model
                 batch_size,
                 seq_length,
                 num_epochs,
                 # save_every,
                 grad_clip,
                 learning_rate,
                 decay_rate,
                 # output_keep_prob
                 input_keep_prob,
                 # init_from,
                 sampling_mode,
                 ):

        self.cuda = cuda
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        # self.rnn_model  =rnn_model
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_epochs = num_epochs
        # self.save_every = save_every
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        # self.output_keep_prob = output_keep_prob
        self.input_keep_prob = input_keep_prob
        # self.init_from = init_from
        self.sampling_mode = sampling_mode

        self.vocab_size = None
