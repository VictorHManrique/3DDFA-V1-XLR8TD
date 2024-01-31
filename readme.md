# Towards Real-Time Lightweight Facial Reconstruction Model

<!-- By [Jianzhu Guo](https://guojianzhu.com/aboutme.html). -->
By [Victor Hernández-Manrique, Miguel González-Mendoza, EugeniaVirtualHumans](https://www.eugenia.tech/).

**\[Updates\]**
 - `2022.5.14`: Recommend a python implementation of face profiling: [face_pose_augmentation](https://github.com/hhj1897/face_pose_augmentation).
 - `2020.8.30`: The pre-trained model and code of ECCV-20 are made public on [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2), the copyright is explained by Jianzhu Guo and the CBSR group.
 - `2020.8.2`: Update a <strong>[simple c++ port](./c++/readme.md)</strong> of this project.
 - `2020.7.3`: The extended work <strong>[Towards Fast, Accurate and Stable 3D Dense Face Alignment](https://guojianzhu.com/assets/pdfs/3162.pdf)</strong> is accepted by [ECCV 2020](https://eccv2020.eu/). See [my page](https://guojianzhu.com) for more details.
 - `2019.9.15`: Some updates, see the commits for details.
 - `2019.6.17`: Adding a [video demo](./video_demo.py) contributed by [zjjMaiMai](https://github.com/zjjMaiMai).
 - `2019.5.2`: Evaluating inference speed on CPU with PyTorch v1.1.0, see [here](#CPU) and [speed_cpu.py](./speed_cpu.py).
 - `2019.4.27`: A simple render pipeline running at ~25ms/frame (720p), see [rendering.py](demo@obama/rendering.py) for more details.
 - `2019.4.24`: Providing the demo building of obama, see [demo@obama/readme.md](demo@obama/readme.md) for more details.
 - `2019.3.28`: Some updates.
 - `2018.12.23`: **Add several features: depth image estimation, PNCC, PAF feature and obj serialization.** See `dump_depth`, `dump_pncc`, `dump_paf`, `dump_obj` options for more details.
 - `2018.12.2`: Support landmark-free face cropping, see `dlib_landmark` option.
 - `2018.12.1`: Refine code and add pose estimation feature, see [utils/estimate_pose.py](./utils/estimate_pose.py) for more details.
 - `2018.11.17`: Refine code and map the 3d vertex to original image space.
 - `2018.11.11`: **Update end-to-end inference pipeline: infer/serialize 3D face shape and 68 landmarks given one arbitrary image, please see readme.md below for more details.**
 - `2018.10.4`: Add Matlab face mesh rendering demo in [visualize](./visualize).
 - `2018.9.9`: Add pre-process of face cropping in [benchmark](./benchmark).

## Introduction

## Applications & Features
#### 1. Face Alignment
<p align="center">
  <img src="samples/dapeng_3DDFA_trim.gif" alt="dapeng">
</p>

#### 2. Face Reconstruction
<p align="center">
  <img src="samples/5.png" alt="demo" width="750px">
</p>

#### 3. 3D Pose Estimation
<p align="center">
  <img src="samples/pose.png" alt="tongliya" width="750px">
</p>

#### 4. Depth Image Estimation
<p align="center">
  <img src="samples/demo_depth.jpg" alt="demo_depth" width="750px">
</p>

#### 5. PNCC & PAF Features
<p align="center">
  <img src="samples/demo_pncc_paf.jpg" alt="demo_pncc_paf" width="800px">
</p>

## Getting started
### Requirements
 - PyTorch >= 0.4.1 (**PyTorch v1.1.0** is tested successfully on macOS and Linux.)
 - Python >= 3.6 (Numpy, Scipy, Matplotlib)
 - Dlib (Dlib is optionally for face and landmarks detection. There is no need to use Dlib if you can provide face bouding bbox and landmarks. Besides, you can try the two-step inference strategy without initialized landmarks.)
 - OpenCV (Python version, for image IO operations.)
 - Cython (For accelerating depth and PNCC render.)
 - Platform: Linux or macOS (Windows is not tested.)

 ```
 # installation structions
 sudo pip3 install torch torchvision # for cpu version. more option to see https://pytorch.org
 sudo pip3 install numpy scipy matplotlib
 sudo pip3 install dlib==19.5.0 # 19.15+ version may cause conflict with pytorch in Linux, this may take several minutes. If 19.5 version raises errors, you may try 19.15+ version.
 sudo pip3 install opencv-python
 sudo pip3 install cython
 ```

In addition, I strongly recommend using Python3.6+ instead of older version for its better design.

### Usage

1. Clone this repo (this may take some time as it is a little big)
    ```
    git clone https://github.com/cleardusk/3DDFA.git  # or git@github.com:cleardusk/3DDFA.git
    cd 3DDFA
    ```

   Then, download dlib landmark pre-trained model in [Google Drive](https://drive.google.com/open?id=1kxgOZSds1HuUIlvo5sRH3PJv377qZAkE) or [Baidu Yun](https://pan.baidu.com/s/1bx-GxGf50-KDk4xz3bCYcw), and put it into `models` directory. (To reduce this repo's size, I remove some large size binary files including this model, so you should download it : ) )


2. Build cython module (just one line for building)
   ```
   cd utils/cython
   python3 setup.py build_ext -i
   ```
   This is for accelerating depth estimation and PNCC render since Python is too slow in for loop.
   
    
3. Run the `main.py` with arbitrary image as input
    ```
    python3 main.py -f samples/test1.jpg
    ```
    If you can see these output log in terminal, you run it successfully.
    ```
    Dump tp samples/test1_0.ply
    Save 68 3d landmarks to samples/test1_0.txt
    Dump obj with sampled texture to samples/test1_0.obj
    Dump tp samples/test1_1.ply
    Save 68 3d landmarks to samples/test1_1.txt
    Dump obj with sampled texture to samples/test1_1.obj
    Dump to samples/test1_pose.jpg
    Dump to samples/test1_depth.png
    Dump to samples/test1_pncc.png
    Save visualization result to samples/test1_3DDFA.jpg
    ```

    Because `test1.jpg` has two faces, there are two `.ply` and `.obj` files (can be rendered by Meshlab or Microsoft 3D Builder) predicted. Depth, PNCC, PAF and pose estimation are all set true by default. Please run `python3 main.py -h` or review the code for more details.

4. Additional example

    ```
    python3 ./main.py -f samples/emma_input.jpg --bbox_init=two --dlib_bbox=false
    ```

<p align="center">
  <img src="samples/emma_input_3DDFA.jpg" alt="samples" width="750px">
</p>

<p align="center">
  <img src="samples/emma_input_pose.jpg" alt="samples" width="750px">
</p>

