## Data Folder

This folder contains the data used for the project. The data is stored in the `data` folder. 

In this folder, the following files exist:<br>
* `kin400.py` --- this file is used to load the data from the `kinetics-400` dataset, inspired from [KineticX Downloader](https://github.com/chi0tzp/KineticX-Downloader) repository. To run this file, run the following command in the the **`code/data/`** directory of this repository (where this `README.md` is located):
```bash
python3 kin400.py
```

* `normalize.py` --- this file takes a video and processes it by resizing it to a 16 x 224 x 224 x 3 tensor.
