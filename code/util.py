import scipy
import numpy as np
import skvideo.io
import torch
import matplotlib.pyplot as plt

#class methods inspired from imported ST-MAE repository
class VideoProcessor:
  num_frames: int
  dim_width:  int
  dim_height: int
  mean: float = 0.45  #from Kinetic400 paper
  std: float  = 0.225 #from Kinetic400 paper

  def __init__(self, num_frames=16, dim_width=224, dim_height=224) -> None:
    self.num_frames = num_frames
    self.dim_width  = dim_width
    self.dim_height = dim_height

  def resize_video(self, video) -> np.ndarray:
    new_frames = self.num_frames / video.shape[0]
    new_width  = self.dim_width / video.shape[1]
    new_height = self.dim_height / video.shape[2]
    new_video = scipy.ndimage.zoom(
        video, 
        (new_frames, new_width, new_height, 1), 
        order=1
    )

    return new_video

  def normalize_frames(self, frames) -> np.ndarray:
    frames = np.float32(frames) / 255
    frames = (frames - self.mean)/self.std

    return frames

  def process_video(self, filepath) -> torch.Tensor:
    frames = skvideo.io.vread(filepath)
    frames = self.resize_video(frames)
    frames = self.normalize_frames(frames)
    frames = torch.tensor(frames).permute(3, 0, 1, 2)
    #frames = frames.uint8

    return frames

  def plot_input(self, tensor) -> None:
    tensor = tensor.float()
    f, axs = plt.subplots(nrows=tensor.shape[0], ncols=tensor.shape[1], figsize=(20, 6))

    tensor = tensor * torch.tensor((self.std, self.std, self.std)).view(3, 1, 1)
    tensor = tensor + torch.tensor((self.mean, self.mean, self.mean)).view(3, 1, 1)

    tensor = torch.clip(tensor * 255, 0, 255).int()

    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            axs[i][j].axis("off")
            axs[i][j].imshow(tensor[i][j].permute(1, 2, 0))
    
    plt.show()

  def plot_video(self, video) -> None:
    #4 elements with diff dimensions: 
    # [frames, height, width, colors=3]
    video = video.permute(1,2,3,0)
    plt.figure(figsize = [24, 24])
    
    #plot frames
    num_frames = video.shape[0]
    for frame_index in range(num_frames):
      plt.subplot(num_frames//4 + 1, 4, frame_index+1)
      #unnormalizing frame
      plt.imshow(torch.clip((video[frame_index]*self.std + self.mean) * 255, 0, 255).int())
      plt.axis('off')

    plt.show()