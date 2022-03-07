class HParams(object):
    """ Hparams was removed from tf 2.0alpha so this is a placeholder
    """

    def __init__(self, **kwargs):
        self.set_defaults()
        self.__dict__.update(kwargs)

    def set_defaults(self):
        self.win_length_ms = 5 ##window len in ms
        self.hop_length_ms = 1 ##shuffle
        self.n_fft = 1024 ##number of frequency bands
        self.ref_level_db = 20 ## decibel scaling of spec
        self.min_level_db = -60 ## noise floor
        self.preemphasis = 0.97 
        self.num_mel_bins = 64 ##mel scaling: frequency scaling that matches cochlea (how compressed)
        self.mel_lower_edge_hertz = 200 ## 500 - 3000 for butcherbird
        self.mel_upper_edge_hertz = 15000 ##
        self.power = 1.5  # for spectral inversion
        self.griffin_lim_iters = 50
        self.butter_lowcut = 500 ##bandpass filter
        self.butter_highcut = 15000 ##bandpass filter (same as mel)
        self.reduce_noise = False
        self.noise_reduce_kwargs = {}
        self.mask_spec = False
        self.mask_spec_kwargs = {"spec_thresh": 0.9, "offset": 1e-10}
        self.nex = -1
        self.n_jobs = -1
        self.verbosity = 1

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
