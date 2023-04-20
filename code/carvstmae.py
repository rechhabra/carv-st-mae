import numpy as np
from util import VideoProcessor
from imported_repos.mae import models_mae
import torch
import tqdm
import sys
import pandas as pd
import scipy
from matplotlib import pyplot as plt

import cv2

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
    top_chunks: int  # we will use the top X chunks to reconstruct the video

    def __init__(self,
                 num_chunk_sampling: int = 500,
                 mask_ratio: int = 0.95,
                 chunk_dim: np.ndarray = [2, 16, 16], #this dim'n is used in ST MAE model
                 top_chunks: int = 150,
                 refresh_rate: int = 10,
                 video_fps: int = 10 #this FPS is used in ST MAE model
                 ):
        self.num_chunk_sampling = num_chunk_sampling
        self.mask_ratio = mask_ratio
        self.chunk_dim = chunk_dim
        self.top_chunks = top_chunks
        self.refresh_rate = refresh_rate
        self.video_fps = video_fps

        self.model = mae_vit_large_patch16(
            decoder_embed_dim=512, decoder_depth=4, mask_type="st", t_patch_size=2)

    def compress_video(self, video_path: str, output_video_name: str = "output.mp4") -> None:
        """
        Compresses the video at the given path using the CARV-ST-MAE algorithm
        """
        video = self.video_processor.process_video(video_path)
        mse = self.generate_chunks_mse(video)
        self.generate_carv_video(mse, video_path, output_video_name)

    def generate_chunks_mse(self, video: torch.Tensor, save_as_csv=False) -> np.ndarray:
        """
        Generates the MSE for each chunk in the video using random mask sampling
        param video: the video to be compressed
        return: the MSE for each chunk in the video
        NOTE: the shape of the returned array is (num_frames/chunk_dim[0], width/chunk_dim[1], height/chunk_dim[2])
        """
        vid_channels, vid_frames, vid_width, vid_height = video.shape

        try:
            mask_frames, mask_width, mask_height = vid_frames / \
                self.chunk_dim[0], vid_width / \
                self.chunk_dim[1], vid_height/self.chunk_dim[2]
            assert mask_frames.is_integer() and mask_width.is_integer() and mask_height.is_integer()
            mask_frames, mask_width, mask_height = int(
                mask_frames), int(mask_width), int(mask_height)
        except AssertionError as e:
            print(
                f"Error: Selected chunk dimensions {self.chunk_dim} do not divide evenly into video dimensions {video.shape}")

        mse_sum: torch.Tensor = torch.tensor(
            np.zeros((mask_frames, mask_width, mask_height)))
        mask_count: np.ndarray = np.zeros(
            (mask_frames, mask_width, mask_height))

        for i in tqdm(range(self.num_chunk_sampling)):
            loss, pred, mask, vis = self.model(
                video.unsqueeze(0), 1, mask_ratio=self.mask_ratio)
            fragment = 1 - (mask.reshape(mask_frames, mask_width, mask_height))
            mask_count = np.add(mask_count, fragment)
            mse_sum = np.add(mse_sum, fragment * (float)(loss))

        # get chunks that were not evaluated
        unnused_chunks = mask_count == 0

        # ensure all mask_count values are not 0
        mask_count = np.where(unnused_chunks, 1, mask_count)
        mse_sum = np.divide(mse_sum, mask_count)

        # save file as a csv, using only the non-ignored elements
        if save_as_csv:
          mse_sum_df = pd.Series(mse_sum.flatten())
          # get elements that were unused -- the MSEs of 0
          unused_chunks = mse_sum_df <= 0
          mse_sum_df = mse_sum_df[~unused_chunks]
          mse_sum_df.to_csv(
              f"mse_sum_{self.num_chunk_sampling}_{self.mask_ratio}.csv")

        return mse_sum

    def normalize_mse_sum(self, mse_sum: np.ndarray):
        # Making MSE values between 0 and 1
        mse_sum = mse_sum - mse_sum.min()
        mse_sum = mse_sum / mse_sum.max()

        mse_sum_norm = mse_sum / mse_sum.mean()
        return mse_sum_norm

    def generate_carv_video(self, mse_sum: np.ndarray, video_filepath: str, output_video_name: str) -> None:
        """
        Method generates a CARV video using the MSE values for each chunk,
        utilizing a downsampling technique

        Input: mse_sum - the MSE values for each chunk in the video
        NOTE: the shape of the input array is (num_frames/chunk_dim[0], width/chunk_dim[1], height/chunk_dim[2])
        video_filepath - the filepath of the video to be compressed

        Output: the compressed video, in mp4 format
        """
        mse_sum_norm = self.normalize_mse_sum(mse_sum)

        mse_num_frames, mse_width, mse_height = mse_sum_norm.shape

        # On average, we expect at every N frames equals a full screen refresh (N = refresh_rate)
        # similar to how many frames to skip in downsampling
        mse_sum_norm /= self.refresh_rate

        cap = cv2.VideoCapture(video_filepath)

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Get video dimensions. Note, depending on video codac it can be buggy
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"Video frame details: {frame_width} width, {frame_height} height, {video_num_frames} frames")

        # Prev_frame is an array containing the last printed screen for every pixel
        prev_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        # prev_increment is a counter for each xy chunk region. When it reaches one
        # The associated chunk will load in. Every loop it increases.
        prev_increment = np.ones((mse_height, mse_width))

        out = cv2.VideoWriter(
            output_video_name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.video_fps,
            (frame_width, frame_height)
        )

        # Read until video is completed
        frame_count = 0
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # Start frame
                next_frame = np.copy(prev_frame)

                # Increment counter
                losses = mse_sum_norm[mse_num_frames *
                                      frame_count//video_num_frames, :, :]
                prev_increment += losses

                # For each chunk region
                for mse_width_ind in range(mse_width):
                    left = mse_width_ind * frame_width // mse_width
                    right = (mse_width_ind+1) * frame_width // mse_width
                    for mse_height_ind in range(mse_height):
                        top = mse_height_ind * frame_height // mse_height
                        bottom = (mse_height_ind+1) * \
                            frame_height // mse_height

                        # If counter triggers, update region,
                        # inspired from the Breshenham's line algorithm
                        if prev_increment[mse_width_ind][mse_height_ind] >= 1:
                            prev_increment[mse_width_ind][mse_height_ind] -= 1
                            next_frame[left:right,
                                       top:bottom] = frame[left:right, top:bottom]
                            prev_frame[left:right,
                                       top:bottom] = frame[left:right, top:bottom]

                # Write video frame
                out.write(next_frame)

                # Increment frame counter
                frame_count += 1

            else:
                break

        # When everything done, release the video capture object
        cap.release()
        out.release()

        # Closes all the frames
        cv2.destroyAllWindows()

        #load video and save as gif
#        clip = VideoFileClip(output_video_name)
#        clip.write_gif(output_video_name.replace(".mp4", ".gif"))



    def plot_chunks_mse_heat_map(self, mse_sum, video):
        mse_sum_df = pd.Series(mse_sum.flatten())

        # filter out chunks that were not selected -- values that are 0
        ignore = mse_sum_df <= 0

        mse_sum_df[ignore] = mse_sum_df.max()+1
        best_chunks = np.argsort(mse_sum_df.ravel())[:self.top_chunks]
        mse_sum_df[ignore] = mse_sum_df.min()-1
        worst_chunks = np.argsort(mse_sum_df.ravel())[-self.top_chunks:]

        best_fragments = np.zeros(mse_sum.shape)
        for mse in best_chunks:
            best_fragments[np.unravel_index((int)(mse), mse_sum.shape)] = 1

        worst_fragments = np.zeros(mse_sum.shape)
        for mse in worst_chunks:
            worst_fragments[np.unravel_index((int)(mse), mse_sum.shape)] = 1

        best_fragments = scipy.ndimage.zoom(
            best_fragments, (self.chunk_dim[0], self.chunk_dim[1], self.chunk_dim[2]), mode="nearest", order=1)
        worst_fragments = scipy.ndimage.zoom(
            worst_fragments, (self.chunk_dim[0], self.chunk_dim[1], self.chunk_dim[2]), mode="nearest", order=1)

        heat_video = 0.25 + video/2
        heat_video[0] = np.add(heat_video[0], 3*worst_fragments)
        heat_video[1] = np.add(heat_video[1], 3*best_fragments)

        self.video_processor.plot_video(
            heat_video,
            title=f"Most important (green) and least {self.top_chunks} chunks (red)\nusing {self.num_chunk_sampling} samples with a {self.mask_ratio} mask ratio"
        )

    def plot_chunks_mse_histogram(self, mse_sum=None, csv_filepath=None):
      mse_sum_df = None
      if csv_filepath != None:
        mse_sum_df = pd.read_csv(csv_filepath)
      elif mse_sum != None:
        mse_sum_df = pd.Series(mse_sum.flatten())
        # get elements that were unused -- the MSEs of 0
        unused_chunks = mse_sum_df <= 0
        mse_sum_df = mse_sum_df[~unused_chunks]
      else:
        raise Exception("Need to pass in either mse_sum or csv_filepath")

      # formatting it
      mse_sum_df.columns = ["ChunkId", "MSE"]

      # ensure it has non-zero MSE values
      mse_sum_df = mse_sum_df[mse_sum_df["MSE"] != 0]

      plt.hist(mse_sum_df["MSE"], bins=100, color='black')
      plt.show()
