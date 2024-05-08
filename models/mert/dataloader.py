import os
from PIL import Image
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# csv_path = r'/home/houdaifa.atou/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/hdf/Darija_Emotion/cross_val.csv'
IMG_SIZE = 224
RESAMPLE_RATE = 16000

def read_frames(path, num_frames):
    entries = list(os.scandir(path))
    # XXX: Sorting for consistency
    entries.sort(
        key=lambda entry: int(entry.name.split('_')[-1].split('.')[0]))
    
    # getting just the number of frames we're intereseted in
    step_size = len(entries) // num_frames
    entries = entries[::step_size][:num_frames]

    assert len(entries)==num_frames


    frames = []
    for entry in entries:
        try:
            p = Image.open(entry.path)
        except Exception as e:
            # TODO: More specific exception
            print(f'Exception {e} while reading frame at: {path}')
        else:
            frames.append(p)
    return frames


def read_audio(path, target_length, resample_rate=RESAMPLE_RATE):
    wave, sample_rate = torchaudio.load(path)
    wave = torchaudio.functional.resample(wave, sample_rate, resample_rate)
    if len(wave.shape) > 1:
        if wave.shape[0] == 1:
            wave = wave.squeeze()
        else:
            wave = wave.mean(axis=0)  # multiple channels, average

    if wave.size(0) > target_length:
        wave = wave[:, :target_length]
    else:
        padding = torch.zeros(target_length - wave.size(0))
        wave = torch.cat((wave, padding), dim=0)
    return wave

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=IMG_SIZE):
        self.img_size = (img_size, img_size)

    def __call__(self, sample):
        trsfrm = transforms.Compose([
            transforms.Resize(size=self.img_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float)
        ])
        sframes = torch.stack([trsfrm(frame) for frame in sample])
        return sframes


class LazyMultiDataset(Dataset):
    def __init__(self, data, num_frames, audio_target_length, transform=None):
        super().__init__()
        self.data = data
        self.num_frames = num_frames
        self.audio_target_length = audio_target_length
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Reading frames
        frames_path = self.data[idx][0]
        frames = read_frames(frames_path, self.num_frames)
        if self.transform:
            frames = self.transform(frames)

        # Reading preprocessed audio
        audio_path = self.data[idx][1]
        audio = read_audio(
            audio_path,
            self.audio_target_length)

        # Reading annotation
        anno = self.data[idx][2]
        sample = {
            'frames': frames,
            'audio': audio,
            'annotations': anno}
        return sample


class LazyMultiData:
    def __init__(self, path, num_frames, audio_target_length) -> None:
        self.path = path
        self.num_frames = num_frames
        self.audio_target_length = audio_target_length
        self.df = pd.read_csv(
            os.path.join(self.path, 'annotations', 'annotations_64.csv'))
        self.frames_path = os.path.join(self.path, 'frames') # 'faces_sep'
        self.audios_path = os.path.join(self.path, 'audios')
        self.transform = ToTensor()
        # self.audio_path = os.path.join(self.path, 'Pre_Audio_5_16')

        # TODO: Assert number of frames

    def get_data(self, split):
        data = []
        split_data = self.df[self.df['split'].isin(split)][['name', 'emotion']]
        split_data = split_data.to_numpy().tolist()

        data = [
            (os.path.join(self.frames_path, items[0]), 
             os.path.join(self.audios_path, f'{items[0]}.wav'),
             items[1]) for items in split_data
        ]
        return data

    def get_dataset(self, split, debug=False):
        data = self.get_data(split)
        if debug:
            # Load just few samples for debugging purposes
            data = data[:10]
        dataset = LazyMultiDataset(
            data=data, 
            num_frames=self.num_frames,
            audio_target_length=self.audio_target_length,
            transform=self.transform)
        print(f'Dataset loaded {len(dataset)}')
        return dataset

    def get_dataloader(self, dataset, shuffle, batch_size=8, num_workers=4):
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True)
        return loader


def main():
    dataset_path = r'/home/houdaifa.atou/lustre/data_sec-um6p-st-sccs-6sevvl76uja/IDS/hdf/Darija_Emotion/'
    dataset = LazyMultiData(dataset_path, 32)
    train_dataset = dataset.get_dataset(split=[1], debug=True)
    dataloader = dataset.get_dataloader(
        train_dataset,
        shuffle=True,
        batch_size=2,
        num_workers=1)
    for data in dataloader:
        audio = data['audio']
        annotations = data['annotations']
        print(audio.shape, annotations.shape)
        break


if __name__ == '__main__':
    main()