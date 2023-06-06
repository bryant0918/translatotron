

class mapDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def create_hparams(hparams_string=None, verbose=False):
    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "epochs": 5000,
        "iters_per_checkpoint": 200,  # Can maybe raise this to like 1000
        "seed": 1234,
        "dynamic_loss_scaling": True,
        "fp16_run": False,
        "distributed_run": True,
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "ignore_layers": ['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        "load_mel_from_disk": False,
        "training_files": 'data/train',
        "validation_files": 'data/val',

        ################################
        # Audio Parameters             #
        ################################
        "max_wav_value": 1.0,  # If using a different package than soundfile you may have to set to 32768
        "sampling_rate": 22050,
        "filter_length": 1024,
        "hop_length": 512,  # originally 256
        "win_length": 1024,
        "n_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,

        ################################
        # Data Parameters             #
        ################################

        # My Computer Paths
        "input_data_root": 'C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final '
                           'Project\\Data\\LibriS2S\\DE',
        "output_data_root": 'C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final '
                            'Project\\Data\\LibriS2S\\EN',
        "data_alignments_csv": 'C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final '
                               'Project\\Data\\LibriS2S\\alignments\\all_de_en_aligned.csv',

        # Super Computer Paths
        # "input_data_root": "/home/bmcarth4/Final Project/Data/LibriS2S/DE",
        # "output_data_root": '/home/bmcarth4/Final Project/Data/LibriS2S/EN',
        # "data_alignments_csv": '/home/bmcarth4/Final Project/Data/LibriS2S/alignments/all_de_en_aligned.csv',

        "train_size": 0.8,
        "test_size": .1,
        # Output Audio Parameters
        "out_channels": 1025,
        ################################
        # Model Parameters             #
        ################################

        # Encoder parameters
        "encoder_kernel_size": 5,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 128,

        # Decoder parameters
        "n_frames_per_step": 1,  # currently only 1 is supported
        "decoder_rnn_dim": 256,
        "prenet_dim": 32,
        "max_decoder_steps": 1000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,

        # Attention parameters
        "attention_rnn_dim": 256,
        "attention_dim": 128,
        "attention_heads": 4,

        # Location Layer parameters
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,

        # Mel-post processing network parameters
        "postnet_embedding_dim": 128,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 2,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate": False,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "grad_clip_thresh": 1.0,
        "batch_size": 4,   # Change to 1 when testing locally, 4 on GPU
        "mask_padding": True
        # set model's padded outputs to padded values
    }

    hparams = mapDict(hparams)

    return hparams
