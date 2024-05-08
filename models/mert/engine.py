"""Training agent"""
import os
import yaml
import math
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# Logging parameters
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

# Configuration builder


class ConfigBuilder:
    def __init__(self, name, required_attributes=None) -> None:
        self.name = name
        if not required_attributes:
            self._required_attributes = []
        else:
            self._required_attributes = required_attributes

    def set_attributes(self, **kwargs):
        # Verify that all required attributes exist
        for ra in self._required_attributes:
            if ra not in kwargs:
                raise ValueError(f'{self.name}: Attirbute {ra} is required')

        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def load_from_yaml(self, yaml_config_file):
        if not os.path.exists(yaml_config_file):
            raise FileNotFoundError(f"Can't find {yaml_config_file}")
        if not yaml_config_file.endswith(('yaml', 'yml')):
            raise ValueError("File extension must be 'yaml' or 'yml'.")
        with open(yaml_config_file) as f:
            model_config = yaml.load(f.read(), Loader=yaml.Loader)
        return self.set_attributes(**model_config)

    def get_attributes(self):
        attrs = vars(self).copy()
        del attrs['_required_attributes']
        del attrs['name']
        return attrs


class TrainConfig(ConfigBuilder):
    def __init__(self, name, required_attributes=None):
        super().__init__(name, required_attributes)
        if 'out_path' not in self._required_attributes:
            self._required_attributes.append('out_path')
        self.out_path = None
        self.checkpoint_path = None

    def set_attributes(self, **kwargs):
        super().set_attributes(**kwargs)
        self.out_path = self.get_out_path(self.out_path)
        self.checkpoint_path = self.get_checkpoint_path()
        return self

    def get_attributes(self):
        attrs = super().get_attributes()
        del attrs['checkpoint_path']
        return attrs

    def get_out_path(self, out_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
            logger.info(f'Folder {out_path} created')
            out_path = os.path.join(out_path, 'version_0')
            os.mkdir(out_path)
            return out_path
        # Get last version
        last_nv = 0
        for entry in os.scandir(out_path):
            if entry.is_dir():
                if entry.name.startswith('version'):
                    try:
                        nv = int(entry.name.removeprefix('version_'))
                    except ValueError:
                        logger.exception(
                            'Not expected behaviour of folder hierarchy')
                    else:
                        if nv > last_nv:
                            last_nv = nv
        out_path = os.path.join(out_path, f'version_{last_nv+1}')
        os.mkdir(out_path)
        return out_path

    def get_checkpoint_path(self):
        checkpoint_path = os.path.join(self.out_path, 'checkpoints')
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        return checkpoint_path

    def set_optimizer(self, optim):
        self.optimizer = optim.__class__.__name__

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler.__class__.__name__


class Config:
    def __init__(self) -> None:
        self.model = ConfigBuilder('Model config')
        self.train = TrainConfig('Train config')
        self.data = ConfigBuilder('Data config')

    def save(self, name='args.yaml'):
        out_dir = self.train.out_path
        out_path = os.path.join(out_dir, name)
        # Remove '_required_attributes' from attributes
        model_attr = self.model.get_attributes()
        train_attr = self.train.get_attributes()
        data_attr = self.data.get_attributes()

        with open(out_path, 'w') as file:
            yaml.dump({
                'model': model_attr,
                'train': train_attr,
                'data': data_attr
            }, file)

    def to_dict(self):
        model_attr = self.model.get_attributes()
        train_attr = self.train.get_attributes()
        data_attr = self.data.get_attributes()

        return {
            'model': model_attr,
            'train': train_attr,
            'data': data_attr
        }

    @classmethod
    def load_from_yaml(cls, yaml_config_file):
        # Verify config file
        if not os.path.exists(yaml_config_file):
            raise FileNotFoundError(f"Can't find {yaml_config_file}")
        if not yaml_config_file.endswith(('yaml', 'yml')):
            raise ValueError("File extension must be 'yaml' or 'yml'.")

        # Read config dictionary
        with open(yaml_config_file) as f:
            config_dict = yaml.load(f.read(), Loader=yaml.Loader)

        # Verify config dictionary values
        if 'model' not in config_dict:
            raise ValueError("Model config is missing")
        if 'train' not in config_dict:
            raise ValueError("Train config is missing")
        if 'data' not in config_dict:
            raise ValueError("Data config is missing")

        # Build config
        conf = Config()
        conf.model = conf.model.set_attributes(**config_dict['model'])
        conf.train = conf.train.set_attributes(**config_dict['train'])
        conf.data = conf.data.set_attributes(**config_dict['data'])
        return conf


class TrainClock:
    # Credit: github.com/henryxrl
    """ Clock object to track epoch and step during training """

    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def load_state_dict(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']

# Metrics


class AverageMeter:
    # Credit: github.com/henryxrl
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# Trainer
        
def tri_stage_lr_scheduler(
        optimizer,
        base_lr,
        total_steps,
        phase_ratio=None,
        init_lr_scale=0.01,
        final_lr_scale=0.01,
        final_decay='exp'):
    """
    Create a learning rate scheduler with warm-up, constant, and linear decay phases.

    Parameters:
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is used.
        warmup_steps (int): Number of warm-up steps.
        total_steps (int): Total number of training steps.
        base_lr (float): Initial learning rate.
        final_lr (float): Final learning rate after linear decay.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Learning rate scheduler.
    """

    if phase_ratio is None:
        phase_ratio = [0.1, 0.4, 0.5]

    assert sum(phase_ratio) == 1, "phase ratios must add up to 1"

    peak_lr = base_lr
    init_lr = init_lr_scale * base_lr
    final_lr = final_lr_scale * base_lr

    warmup_steps = int(total_steps * phase_ratio[0])
    hold_steps = int(total_steps * phase_ratio[1])
    decay_steps = int(total_steps * phase_ratio[2])

    warmup_rate = (
        (peak_lr - init_lr) / warmup_steps
        if warmup_steps != 0
        else 0
    )

    def decay_factor(steps_in_stage):
        if final_decay == 'exp':
            decay_rate = -math.log(final_lr_scale) / decay_steps
            factor = math.exp(-decay_rate * steps_in_stage)
            return factor
        elif final_decay == 'linear':
            decay_rate = (
                (final_lr - peak_lr) / decay_steps
                if decay_steps != 0
                else 0
            )
            factor = 1 + (decay_rate * steps_in_stage) / base_lr
            return factor
        elif final_decay == 'cosine':
            decay_rate = (
                steps_in_stage / decay_steps
                if decay_steps != 0
                else 0
            )
            factor = final_lr_scale + \
                (1 - final_lr_scale) * 0.5 * (1. + math.cos(math.pi * decay_rate))
            return factor
        else:
            raise NotImplementedError

    def _decide_stage(update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < warmup_steps:
            # warmup state
            return 0, update_step

        offset = warmup_steps

        if update_step < offset + hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += hold_steps

        if update_step <= offset + decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset

    def lr_lambda(step):
        """Update the learning rate after each update."""
        stage, steps_in_stage = _decide_stage(step)
        if stage == 0:
            factor = init_lr_scale + (warmup_rate * steps_in_stage) / base_lr
        elif stage == 1:
            factor = 1
        elif stage == 2:
            factor = decay_factor(steps_in_stage)
        elif stage == 3:
            factor = final_lr_scale

        return factor

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, config, device):
        # self.model = model
        self.config = config
        self.device = device
        self.writer = SummaryWriter(log_dir=self.config.out_path)
        self.validate = True if config.val is not None else False


    # Train epoch
    def train_epoch(
            self,
            model,
            clock,
            dataloader,
            optimizer):
        model.train()
        mloss = AverageMeter('loss')
        macc = AverageMeter('acc')
        for data in tqdm(dataloader):
            audio = data['audio'].to(self.device)
            frames = data['frames'].to(self.device)
            annotations = data['annotations'].to(self.device)

            # Zero to parameter gradients
            optimizer.zero_grad()

            # Get predections
            predections = model(audio, frames)

            # Loss
            loss = F.cross_entropy(predections, annotations)

            # Accuracy
            _, predicted = torch.max(predections, 1)
            acc = torch.mean((predicted == annotations).float())

            # Backward and optimizr step
            loss.backward()
            optimizer.step()

            # Saving metrics
            mloss.update(loss.item())
            macc.update(acc.item())

            # Clock
            clock.tick()

        # Log metrics
        if self.writer:
            self.writer.add_scalar('Loss/train', mloss.avg, clock.epoch)
            self.writer.add_scalar('Accuracy/train', macc.avg, clock.epoch)

        logger.info(f'Epoch {clock.epoch} training loss: {mloss.avg}')
        logger.info(f'Epoch {clock.epoch} training acc: {macc.avg}')

        return mloss.avg

    # Define the validation function
    def val_epoch(
            self,
            model,
            clock,
            dataloader):
        model.eval()
        with torch.no_grad():
            mloss = AverageMeter('loss')
            macc = AverageMeter('acc')
            for data in tqdm(dataloader):
                audio = data['audio'].to(self.device)
                frames = data['frames'].to(self.device)
                annotations = data['annotations'].to(self.device)

                # Get predections
                predections = model(audio, frames)

                # Loss
                loss = F.cross_entropy(predections, annotations)

                # Accuracy
                _, predicted = torch.max(predections, 1)
                acc = torch.mean((predicted == annotations).float())

                # Saving metrics
                mloss.update(loss.item())
                macc.update(acc.item())

        # Log metrics
        if self.writer:
            self.writer.add_scalar('Loss/val', mloss.avg, clock.epoch)
            self.writer.add_scalar('Accuracy/val', macc.avg, clock.epoch)

        logger.info(f'Epoch {clock.epoch} validation loss: {mloss.avg}')
        logger.info(f'Epoch {clock.epoch} validation acc: {macc.avg}')
        return mloss.avg, macc.avg

    def save_ckpt(
            self,
            model,
            clock,
            optimizer,
            scheduler,
            name=None):

        if name is None:
            name = f'model_epoch_{clock.epoch + 1}'

        checkpoint_path = os.path.join(
            self.config.checkpoint_path, f'{name}.pt')
        logger.info(f"Saving checkpoint {name} at epoch {clock.epoch + 1}")

        torch.save({
            'clock': clock.make_checkpoint(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, checkpoint_path)

    def write_hparams(self, config):
        config_dict = config.to_dict()
        json_p = dict_to_markdown_table(config_dict)
        self.writer.add_text('Configurations', json_p)

    def load(self, model, clock, optimizer=None, scheduler=None):
        state = torch.load(
            self.config.last_trained_checkpoint, map_location=self.device)
        model.load_state_dict(state['model_state_dict'])
        clock.load_state_dict(state['clock'])
        if optimizer:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(state['scheduler_state_dict'])

    def fit(self, model, dataset):
        # Model
        model.to(self.device)
        logger.info(f'Model device: {next(model.parameters()).device}')

        # Initialize optimizer
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.lr,
            betas=self.config.opt_betas,
            weight_decay=self.config.opt_wd)
        
        scheduler = tri_stage_lr_scheduler(
            optimizer=optimizer,
            base_lr=self.config.lr,
            total_steps=self.config.epochs,
            final_decay=self.config.sched_final_decay,
            phase_ratio=self.config.sched_phase_ration)

        # Training clock
        clock = TrainClock()

        # Setting optimizer and scheduler for debugging
        self.config.set_optimizer(optimizer)
        self.config.set_scheduler(scheduler)

        # Load
        # TODO: Change name
        if self.config.last_trained_checkpoint:
            self.load(model, clock, optimizer, scheduler)

        # Train data loader
        train_dataset = dataset.get_dataset(
            split=self.config.train, debug=self.config.debug)
        train_laoder = dataset.get_dataloader(
            train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers)

        # Vaildation data loader
        if self.validate:
            val_dataset = dataset.get_dataset(
                split=self.config.val, debug=self.config.debug)
            val_loader = dataset.get_dataloader(
                val_dataset,
                shuffle=False,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers)

            minvalloss = np.Inf
            bepochloss = 0

            maxvalacc = -np.Inf
            bepochacc = 0

        # Training loop
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(
                model,
                clock,
                train_laoder,
                optimizer)

            if (epoch + 1) % self.config.save_interval == 0:
                self.save_ckpt(
                    model,
                    clock,
                    optimizer,
                    scheduler)


            if self.validate:
                val_loss, val_acc = self.val_epoch(
                    model, clock, val_loader)
                if minvalloss > val_loss:
                    minvalloss = val_loss
                    bepochloss = epoch
                    self.save_ckpt(
                        model,
                        clock,
                        optimizer,
                        scheduler,
                        'best_loss')

                if maxvalacc < val_acc:
                    maxvalacc = val_acc
                    bepochacc = epoch
                    self.save_ckpt(
                        model,
                        clock,
                        optimizer,
                        scheduler,
                        'best_acc')

            logger.info(f"{'#'*15} Epoch {epoch} done {'#'*15}")

            # Logg learning rate
            self.writer.add_scalar(
                'learning_rate', optimizer.param_groups[-1]['lr'], clock.epoch)

            # Update learning rate
            scheduler.step()
            clock.tock()

        self.save_ckpt(model, clock, optimizer, scheduler)

        if self.validate:
            logger.info(
                f'Best loss {bepochloss}:{minvalloss}, Best acc {bepochacc}:{maxvalacc}')


def dict_to_markdown_table(data):
    table = "|Config|Parameter|Value|\n"
    table += f"| {' | '.join(['---' for _ in range(3)])} |\n"

    for key, value in data.items():
        first_key = list(value)[0]
        table += f"| {key} | {first_key} | {value.pop(first_key)}\n"
        for k, v in value.items():
            if isinstance(v, dict):
                table += f"|   | {k} | {dict_to_str(v)} |\n"
            else:
                table += f"|   | {k} | {v} |\n"
    return table


def dict_to_str(data):
    dict_str = ''
    for k, v in data.items():
        dict_str += f'{k}: {v} <br>'
    return dict_str