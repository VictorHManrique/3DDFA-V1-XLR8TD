# Towards Real-Time Lightweight Facial Reconstruction Model

<!-- By [Jianzhu Guo](https://guojianzhu.com/aboutme.html). -->
By [Victor Hernández-Manrique, Miguel González-Mendoza, EugeniaVirtualHumans](https://www.eugenia.tech/).

**\[Updates\]**
 - `2024.01.31`: 3DDFA-V1 XLR8TD

## Requirements
Anaconda

## Usage

1. Clone this repo
    ```
    git clone https://github.com/VictorHManrique/3DDFA-V1-XLR8TD.git
    ```

    ```
    cd 3DDFA-V1-XLR8TD
    ```

   Then, download dlib landmark pre-trained model in [Google Drive](https://drive.google.com/open?id=1kxgOZSds1HuUIlvo5sRH3PJv377qZAkE) or [Baidu Yun](https://pan.baidu.com/s/1bx-GxGf50-KDk4xz3bCYcw), and put it into `models` directory. (To reduce this repo's size)

2. Create Conda Environment
    ```
    conda env create -f environment.yml
    ```

3. Activate Conda Environment
    ```
    conda activate XLR8TD
    ```

4. Build cython module (just one line for building)
    ```
    cd utils/cython
    ```
    
    ```
    python3 setup.py build_ext -i
    ```
    
5. Return to Main
    ```
    cd ../..
    ```
    
6. Run the UI
    ```
    python3 xlr8td.py
    ```
