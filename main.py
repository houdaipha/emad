import os
import argparse
import torch
import importlib


parser = argparse.ArgumentParser()
parser.add_argument(
    '-conf',
    '--config_path',
    help='path to config file',
    required=True)
parser.add_argument('-m', '--model', help='Model name', required=True)
parser.add_argument(
    '-l',
    '--lightning',
    help='Use pytorch lighting',
    default=False,
    action='store_true')
parser.add_argument(
    '-d',
    '--device_id',
    default="0",
    help='cuda device id',
    type=str)
args = parser.parse_args()


def todevice(device, is_lightning):
    device = [int(item) for item in device.split(',')]
    if is_lightning:
        return device
    if len(device) == 1:
        return device[0]
    return device


# XXX: Importing model like:
# from models import resnetfwlstm
model = importlib.import_module(f'models.{args.model}')

# Mitigate gpu allocation problem
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def main():
    # Device
    if args.lightning:
        # XXX: Pytorch lightning
        device_id = todevice(args.device_id, True)
        model.pl_train(
            config_path=args.config_path,
            devices=device_id)
    else:
        # XXX: Pytorch
        device_id = todevice(args.device_id, False)
        if not isinstance(device_id, int):
            raise TypeError('Device id is not an integer')
        device = torch.device(
            f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        model.train(config_path=args.config_path, device=device)


if __name__ == '__main__':
    main()
