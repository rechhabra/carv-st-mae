from carvstmae import CARVSTMAE
import scipy
import numpy as np
import skvideo.io
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import skvideo.measure

# class methods inspired from imported ST-MAE repository


class VideoProcessor:
    num_frames: int
    dim_width:  int
    dim_height: int
    mean: float = 0.45  # from Kinetic400 paper
    std: float = 0.225  # from Kinetic400 paper

    def __init__(self, num_frames=16, dim_width=224, dim_height=224) -> None:
        self.num_frames = num_frames
        self.dim_width = dim_width
        self.dim_height = dim_height

    def resize_video(self, video) -> np.ndarray:
        new_frames = self.num_frames / video.shape[0]
        new_width = self.dim_width / video.shape[1]
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
        # frames = frames.uint8

        return frames

    def plot_input(self, tensor) -> None:
        tensor = tensor.float()
        f, axs = plt.subplots(
            nrows=tensor.shape[0], ncols=tensor.shape[1], figsize=(20, 6))

        tensor = tensor * \
            torch.tensor((self.std, self.std, self.std)).view(3, 1, 1)
        tensor = tensor + \
            torch.tensor((self.mean, self.mean, self.mean)).view(3, 1, 1)

        tensor = torch.clip(tensor * 255, 0, 255).int()

        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                axs[i][j].axis("off")
                axs[i][j].imshow(tensor[i][j].permute(1, 2, 0))

        plt.show()

    def plot_video(self, video, title="") -> None:
        # 4 elements with diff dimensions:
        # [frames, height, width, colors=3]
        video = video.permute(1, 2, 3, 0)
        plt.figure(figsize=[24, 24])

        # plot frames
        num_frames = video.shape[0]
        for frame_index in range(num_frames):
            plt.subplot(num_frames//4 + 1, 4, frame_index+1)
            # unnormalizing frame
            plt.imshow(torch.clip(
                (video[frame_index]*self.std + self.mean) * 255, 0, 255).int())
            plt.axis('off')

        plt.title(title)
        plt.show()


class CompressionStatisticGenerator:
    """
    This class is to generate videos of a certain compression type
    to compare compression statistics with the original video
    """
    carv_st_mae: CARVSTMAE

    def __init__(self, carv_st_mae: CARVSTMAE = None) -> None:
        self.carv_st_mae = carv_st_mae or CARVSTMAE(mask_ratio=0.98, num_chunk_sampling=500)

    def generate_simulated_video(
            self,
            mse_sum: np.ndarray,
            original_video_filepath: str,
            output_video_name: str,
            simulate_downsampling=False,
            simulate_randomchunkselection=False):
        """
        CARV method: both false (default)
        Naive downsampling (Every Nth frames): downsampling true
        Every Nth frames with random offset: randomchunkselection true
        """

        if (simulate_downsampling == False and simulate_randomchunkselection == False):
            self.carv_st_mae.generate_carv_video(
                mse_sum, original_video_filepath, output_video_name)

        mse_sum = self.carv_st_mae.normalize_mse_sum(mse_sum)

        mse_num_frames, mse_width, mse_height = mse_sum.shape

        # On average, we expect at every N frames equals a full screen refresh (N = refresh_rate)
        # similar to how many frames to skip in downsampling
        mse_sum /= self.carv_st_mae.refresh_rate

        if (simulate_downsampling or simulate_randomchunkselection):
            regdown = np.zeros(mse_sum.shape)
            regdown.fill(mse_sum.mean())
            mse_sum = regdown

        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(original_video_filepath)

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(
            f"Video frame details: {frame_width} width, {frame_height} height, {video_num_frames} frames")

        prev_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        prev_increment = np.ones((mse_height, mse_width))

        if (simulate_randomchunkselection):
            prev_increment = np.random.rand(mse_height, mse_width)
            prev_increment += 1

        out = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(
            *'mp4v'), self.carv_st_mae.video_fps, (frame_width, frame_height))
        # Read until video is completed
        frame_count = 0
        while (cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                next_frame = np.copy(prev_frame)
                losses = mse_sum[mse_num_frames *
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

                    if prev_increment[mse_width_ind][mse_height_ind] >= 1:
                        prev_increment[mse_width_ind][mse_height_ind] -= 1
                        next_frame[left:right,
                                   top:bottom] = frame[left:right, top:bottom]
                        prev_frame[left:right,
                                   top:bottom] = frame[left:right, top:bottom]

                out.write(next_frame)

                frame_count += 1
            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()
        out.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    def generate_compression_statistics(
        self,
        original_video_path: str,
        carv_video_path: str,
        downsample_video_path: str,
        random_chunk_video_path: str
    ):
        original = skvideo.io.vread(original_video_path)
        carv = skvideo.io.vread(carv_video_path)
        naive = skvideo.io.vread(downsample_video_path)
        random = skvideo.io.vread(random_chunk_video_path)

        original_grey = skvideo.io.vread(original_video_path, as_grey=True)
        carv_grey = skvideo.io.vread(carv_video_path, as_grey=True)
        naive_grey = skvideo.io.vread(downsample_video_path, as_grey=True)
        random_grey = skvideo.io.vread(random_chunk_video_path, as_grey=True)

        toCompare = [carv, naive, random]
        toCompare_grey = [carv_grey, naive_grey, random_grey]
        for video in toCompare:
            mse = (np.square(original - video)).mean()
            print("MSE: ", mse)

        for video in toCompare_grey:
            psnr = skvideo.measure.psnr(original_grey, video)
            print("PSNR mean: ", psnr.mean())

        for video in toCompare_grey:
            ssim = skvideo.measure.ssim(original_grey, video)
            print("SSIM mean: ", ssim.mean())

        for video in toCompare_grey:
            strred, final_after, final_before = skvideo.measure.strred(
                original_grey, video)
            print("ST-RRED: ", final_after, "ST-RRED SSN: ", final_before)
            