# Towards Real-Time Lightweight Facial Reconstruction Model

<!-- By [Jianzhu Guo](https://guojianzhu.com/aboutme.html). -->
By [Victor Hernández-Manrique, Miguel González-Mendoza, EugeniaVirtualHumans](https://www.eugenia.tech/).

**\[Updates\]**
 - `2024.01.31`: 3DDFA-V1 XLR8TD

## Usage

1. Create conda environment 
    ```
    conda create --name XLR8TD python=3.11.5
    ```
    
2. Activate conda environment
    ```
    conda activate XLR8TD
    ```
    
2. Clone this repo
    ```
    git clone https://github.com/VictorHManrique/3DDFA-V1-XLR8TD.git
    cd 3DDFA-V1-XLR8TD
    ```

   Then, download dlib landmark pre-trained model in [Google Drive](https://drive.google.com/open?id=1kxgOZSds1HuUIlvo5sRH3PJv377qZAkE) or [Baidu Yun](https://pan.baidu.com/s/1bx-GxGf50-KDk4xz3bCYcw), and put it into `models` directory. (To reduce this repo's size)

3. Install default packages
    ```
    pip install torch torchvision
    pip install numpy scipy matplotlib
    pip install dlib
    pip install opencv-python
    pip install cython
    pip install kivy
    pip install kivymd
    ```

4. Build cython module (just one line for building)
   ```
   cd utils/cython
   python3 setup.py build_ext -i
   ```
   This is for accelerating depth estimation and PNCC render since Python is too slow in for loop.

5. Return to main
   ```
   cd ..
   ```
   
    
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

4. Run the `xlr8td.py`

    ```
    python3 xlr8td.py
    ```

