import numpy as np
import torch
from ..mert.engine import Config, Trainer
from ..mert.model import MERT
from ..mert.dataloader import LazyMultiData


def set_seed(seed, device_type):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in device_type:
        print('Seeding cuda')
        # if using CUDA for GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(config_path, device):
    config = Config.load_from_yaml(config_path)
    if config.train.seed is not None:
        set_seed(config.train.seed, device.type)
    model = MERT(config.model, device)
    data = LazyMultiData(
        config.data.root_dir, 
        config.data.num_frames,
        config.data.audio_target_length)
    trainer = Trainer(config.train, device)
    config.save()
    trainer.write_hparams(config)
    trainer.fit(model, data)
