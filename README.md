# ImageColorization
Forked from [github.com/praywj/Interactive-Deep-Colorization-and-Compression](https://github.com/praywj/Interactive-Deep-Colorization-and-Compression)

This originally was the code for the paper "Interactive Deep Colorization and Its Application for Image Compression" 
by Xiao et al.  
We expanded it with a few methods to choose the local cue points. Including a  method, that worked better than the 
approach used by Xiao et al.  

We also added a GAN Compression system by Agustsson et al., instead of the original BPG compression. 

## Point Picking Methods

## Getting Started

### Prerequisites
- A Linux of choice, for the feeling of superiority.  
- NVIDIA GPU for training, or CPU for running only with at least ~8 GB of RAM  
- Tensorflow 2.X + python  

### Pretrained models
The pretrained models are available here: [models](https://drive.google.com/drive/folders/1yzbB6oZSHxe_WYmUzOuujs2zTXRc2z3M?usp=sharing)  
They were trained on ~100.000 images of the ImageNet train set cropped to 256x256. 


### Preparing for training
The file preprocess.py will *preprocess* the images used for training. For both the colorization and compression.   

Adjust the code in `main()` to your needs before running.  
Call `dataset_prepare(set="train", max_img=100_000)` with the number of images you want to use of this set. This 
will symlink random images into `res/set/original_img`.  
Note: This code was written with the ImageNet Dataset folder structure in mind. For other datasets, you might need 
to adjust the code in other places  as well.  
`preprocess_color` and `preprocess_grayscale` will generate the required data for training, but also for some 
recolorization and compression steps. You can choose what to generate and whether to overwrite as well. The random 
crop uses the filename as a seed, to produce the same crop for color and grayscale images. 

### Run
TODO  

### Training
TODO  

