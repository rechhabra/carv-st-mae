# Algorithmic Video Reduction by Estimating Chunk Significance

![CARV Demo](visualizations/carvstmae_compression_comparison_gif_0.7.gif)

This repository is for the paper "Algorithmic Video Reduction by Estimating Chunk Significance Using Spatial-Temporal Masked Auto Encoders", which is made for the graduate course "Artificial Intelligence" at Boston University.

The project proposal and milestone report are located in the `documents` folder and the code is respectively located in the `code` folder.

**Authors:** <br> Reshab Chhabra, Mark D. Yang

## Installation Tutorial

1. Download the repository and install the dependencies using the following command in the root directory of this repository:
```bash
pip install -r code/requirements.txt
```
2. Download a pre-trained MAE model finetuned to videos (tuned to Kinetics 400 dataset) by running the command **in the `models` folder**:<br>
```bash
wget 'https://dl.fbaipublicfiles.com/video-mae-100x4-joint.pth' -O checkpoint.pth
```
or
```
curl 'https://dl.fbaipublicfiles.com/video-mae-100x4-joint.pth' -O checkpoint.pth
```
Depending on your operating system.

3. See the `code/imported_repos/README.md` file for more information on downloading the dependent, imported repositories.

4. See the `data/README.md` file for more information on downloading the data used in the paper.

## Demo

To see a demonstration of the repo and figures replication, check out the [colab notebook](https://colab.research.google.com/drive/1tG1-e4f5A5SFVvhDvZMGrWLRA_CE4Llb?usp=sharing). The downloaded notebook `carv_st_mae_demo.ipynb` is also conveniently located in this folder.

<br><br><br><br>**Citation**:<br>
Chhabra, R., & Yang, M. D. (2023). Algorithmic Video Reduction by Estimating Chunk Significance Using Spatial-Temporal Masked Auto Encoders. _GitHub_, github.com/rechhabra/carv-st-mae