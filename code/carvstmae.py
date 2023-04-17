import numpy as np
from util import VideoProcessor
from imported_repos.mae import models_mae
import torch
import tqdm
import sys
import pandas as pd
import scipy

sys.path.append('./imported_repos/mae')
sys.path.append('./imported_repos/')
sys.path.append('..')

from models_mae import mae_vit_large_patch16

class CARVSTMAE:
    num_chunk_sampling: int
    mask_ratio: int
    chunk_dim: np.ndarray
    video_processor: VideoProcessor = VideoProcessor()
    model: mae_vit_large_patch16
    top_chunks: int #we will use the top X chunks to reconstruct the video

    def __init__(self,
                 num_chunk_sampling: int = 500,
                 mask_ratio: int = 0.95,
                 chunk_dim: np.ndarray=[2, 16, 16],
                 top_chunks: int = 200
    ):
        self.num_chunk_sampling = num_chunk_sampling
        self.mask_ratio = mask_ratio
        self.chunk_dim = chunk_dim
        self.top_chunks = top_chunks

    def compress_video(self, filepath: str):
        video = self.video_processor.process_video(filepath)
        chunks = self.randomly_sample_chunks(video)
        #TODO: compress video

    def generate_chunks_mse(self, video: torch.Tensor) -> np.ndarray:
        """
        Generates the MSE for each chunk in the video using random mask sampling
        param video: the video to be compressed
        return: the MSE for each chunk in the video
        """
        vid_channels, vid_frames, vid_width, vid_height = video.shape

        try:
            mask_frames, mask_width, mask_height = vid_frames/self.chunk_dim[0], vid_width/self.chunk_dim[1], vid_height/self.chunk_dim[2]
            assert mask_frames.is_integer() and mask_width.is_integer() and mask_height.is_integer()
            mask_frames, mask_width, mask_height = int(mask_frames), int(mask_width), int(mask_height)
        except AssertionError as e:
            print(f"Error: Selected chunk dimensions {self.chunk_dim} do not divide evenly into video dimensions {video.shape}")

        mse_sum: torch.Tensor = torch.tensor(np.zeros((mask_frames, mask_width, mask_height)))
        mask_count: np.ndarray = np.zeros((mask_frames, mask_width, mask_height))

        for i in tqdm.tqdm(range(self.num_chunk_sampling)):
            loss, pred, mask, vis = self.model(video.unsqueeze(0), 1, mask_ratio=self.mask_ratio)
            fragment = 1 - (mask.reshape(mask_frames, mask_width, mask_height))
            mask_count = np.add(mask_count, fragment)
            mse_sum = np.add(mse_sum, fragment * (float)(loss))

        #ensure all mask_count values are not 0
        mask_count = np.where(mask_count == 0, 1, mask_count)

        mse_sum = np.divide(mse_sum, mask_count)

        return mse_sum

        
    def plot_chunks_mse_heat_map(self, mse_sum, video):
        mse_sum_df = pd.Series(mse_sum.flatten())

        #filter out chunks that were not selected -- values that are 0
        mse_sum_df = mse_sum_df[mse_sum_df > 0]

        best_chunks = np.argsort(mse_sum_df.ravel())[:self.top_chunks]
        worst_chunks= np.argsort(mse_sum_df.ravel())[-self.top_chunks:]

        best_fragments = np.zeros(mse_sum.shape)

        for mse in best_chunks:
            best_fragments[np.unravel_index((int)(mse), mse_sum.shape)] = 1

        worst_fragments = np.zeros(mse_sum.shape)
        for mse in worst_chunks:
            worst_fragments[np.unravel_index((int)(mse), mse_sum.shape)] = 1

        best_fragments = scipy.ndimage.zoom(best_fragments, (self.chunk_dim[0], self.chunk_dim[1], self.chunk_dim[2]), mode="nearest", order=1)
        worst_fragments = scipy.ndimage.zoom(worst_fragments, (self.chunk_dim[0], self.chunk_dim[1], self.chunk_dim[2]), mode="nearest", order=1)

        heat_video = 0.25 + video/2
        heat_video[0] = np.add(heat_video[0], 3*worst_fragments)
        heat_video[1] = np.add(heat_video[1], 3*best_fragments)

        self.video_processor.plot_video(
            heat_video, 
            title = f"Most important (green) and least {self.top_chunks} chunks (red)\nusing {self.num_chunk_sampling} samples with a {self.mask_ratio} mask ratio"
        )    