MEAN = (0.45, 0.45, 0.45)
STD = (0.225, 0.225, 0.225)

import scipy
import skvideo.io
import numpy as np
import torch
from matplotlib import pyplot as plt

def resize_video(vid):
    f_f = 16 / vid.shape[0]
    w_f = 224 / vid.shape[1]
    h_f = 224 / vid.shape[2]
    
    vid = scipy.ndimage.zoom(vid, (f_f, w_f, h_f, 1), order=1)
    return vid

#normalizing videos to be 16 x 228 x 228 x 3
def prepare_video(path):
    frames = skvideo.io.vread(path)
    frames = resize_video(frames)
    frames = np.float32(frames) / 255
    frames = (frames - 0.45)/0.225 #Mean and std
    frames = torch.tensor(frames).permute(3, 0, 1, 2)
    #frames = frames.uint8
    return frames

def plot_input(tensor):
    tensor = tensor.float()
    f, ax = plt.subplots(nrows=tensor.shape[0], ncols=tensor.shape[1], figsize=(50, 20))

    tensor = tensor * torch.tensor(STD).view(3, 1, 1)
    tensor = tensor + torch.tensor(MEAN).view(3, 1, 1)
    tensor = torch.clip(tensor * 255, 0, 255).int()

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            ax[i][j].axis("off")
            ax[i][j].imshow(tensor[i][j].permute(1, 2, 0))
    plt.show()

def show_video(video, title=''):
    video = video.permute(1,2,3,0)
    # video is [F, H, W, 3]
    assert video.shape[3] == 3
    plt.figure(figsize=[24,24])
    for i in range(video.shape[0]):
      plt.subplot(video.shape[0]//4 + 1, 4, i + 1)

      plt.imshow(torch.clip((video[i]*0.225 + 0.45) * 255, 0, 255).int())
      #plt.title(title, fontsize=16)
      plt.axis('off')
    return