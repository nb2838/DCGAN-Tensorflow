# e4040-2021Spring-project DCGANS: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

This Readme contains an implementaion of the paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks."  The folder DCGAN contains all of the relevant files for the implementation and the jupyter file can be viewed as interactive report. See below for a thorough explanation. 

# Data
This paper uses the Celeba Dataset and the cifar-10 dataset. The cifar-10 datset is downloaded via the relevant scripts in the jupyter notebook in this directory. However, the celeba needs to be dowloaded manually in  [this link](https://www.kaggle.com/jessicali9530/celeba-dataset) and should be unzipped in the ``data`` folder. 

# Organization of this Directory
This directory contains four main parts 
- ``writeup``: The writeup folder contains a pdf and `.tex` file with the written report for the project. 
- ``final_presentation_notebook.ipynb``: This contains a jupyter notebook with the code for the project. It can be viewed as an interactive version of the report. Everything is runnable without the data except for the training step and the preview of images from the paper. 

- ``DCGAN:`` This folder contains the main code for the project. It contains two main files. One for the training of the cifar neural network and one for the training of celeba neural network (see report for further explanation). I only kept the last neural networks that were trained. In addition this folder contains utilities that were used throughout training and a folder with the checkpoints where the models are stored. I couldn't upload all of the checkpoints because it was too much storage space. 
  
- ``paper_data``: This contains images that are used for the report. Additionally, it contains some of the images that were produced during the training of the neural network which are very interesting to look at. I didn't save all of the images simply for the sake of organization. 
   
- ``requirements.txt``: The requirements.txt file contains all of the requirements to run the notebook. The main difference from the file used in the assignments was that I had to use tensorflow 2.4 due to a conflict in my machine. 

- ``data``: A folder where data should be placed following the instructions above. x.

```
├── DCGAN
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   ├── celebamodels.cpython-37.pyc
│   │   ├── cifar10models.cpython-37.pyc
│   │   └── utils.cpython-37.pyc
│   ├── celeba_checkpoints
│   │   ├── checkpoint
│   │   ├── ckpt-13.data-00000-of-00001
│   │   └── ckpt-13.index
│   ├── celebamodels.py
│   ├── cifar10_checkpoints
│   │   ├── checkpoint
│   │   ├── ckpt-50.data-00000-of-00001
│   │   └── ckpt-50.index
│   ├── cifar10models.py
│   └── utils.py
├── README.md
├── final_presentation_notebook.ipynb
├── paper_data
│   ├── additional_figures
│   │   ├── dcgan_architecture.png
│   │   ├── flowchart.png
│   │   ├── gan_algorithm.png
│   │   ├── generated_faces.png
│   │   ├── image_net_paper_results.png
│   │   ├── input_dropout.png
│   │   ├── my_vector_arithmetic.png
│   │   ├── no_dropout.png
│   │   ├── one_dropout.png
│   │   ├── space_walking.png
│   │   └── vector_arithmetic.png
│   ├── celeba
│   │   ├── brown_hair_man.png
│   │   ├── brown_hair_woman.png
│   │   ├── celeb_generated_images.png
│   │   ├── celeb_sample_images.png
│   │   ├── discriminator.png
│   │   ├── generator.png
│   │   ├── progression_images
│   │   │   ├── progression_image_0_1000.png
│   │   │   ├── progression_image_0_1500.png
│   │   │   ├── progression_image_0_500.png
│   │   │   ├── progression_image_10_1000.png
│   │   │   ├── progression_image_10_1500.png
│   │   │   ├── progression_image_10_500.png
│   │   │   ├── progression_image_11_1000.png
│   │   │   ├── progression_image_11_1500.png
│   │   │   ├── progression_image_11_500.png
│   │   │   ├── progression_image_12_1000.png
│   │   │   ├── progression_image_12_1500.png
│   │   │   ├── progression_image_12_500.png
│   │   │   ├── progression_image_13_1000.png
│   │   │   ├── progression_image_13_1500.png
│   │   │   ├── progression_image_13_500.png
│   │   │   ├── progression_image_14_1000.png
│   │   │   ├── progression_image_14_1500.png
│   │   │   ├── progression_image_14_500.png
│   │   │   ├── progression_image_15_1000.png
│   │   │   ├── progression_image_15_1500.png
│   │   │   ├── progression_image_15_500.png
│   │   │   ├── progression_image_16_1000.png
│   │   │   ├── progression_image_16_1500.png
│   │   │   ├── progression_image_16_500.png
│   │   │   ├── progression_image_17_1000.png
│   │   │   ├── progression_image_17_1500.png
│   │   │   ├── progression_image_17_500.png
│   │   │   ├── progression_image_18_1000.png
│   │   │   ├── progression_image_18_1500.png
│   │   │   ├── progression_image_18_500.png
│   │   │   ├── progression_image_19_1000.png
│   │   │   ├── progression_image_19_1500.png
│   │   │   ├── progression_image_19_500.png
│   │   │   ├── progression_image_1_1000.png
│   │   │   ├── progression_image_1_1500.png
│   │   │   ├── progression_image_1_500.png
│   │   │   ├── progression_image_20_1000.png
│   │   │   ├── progression_image_20_1500.png
│   │   │   ├── progression_image_20_500.png
│   │   │   ├── progression_image_21_1000.png
│   │   │   ├── progression_image_21_1500.png
│   │   │   ├── progression_image_21_500.png
│   │   │   ├── progression_image_22_1000.png
│   │   │   ├── progression_image_22_1500.png
│   │   │   ├── progression_image_22_500.png
│   │   │   ├── progression_image_24_1000.png
│   │   │   ├── progression_image_24_1500.png
│   │   │   ├── progression_image_24_500.png
│   │   │   ├── progression_image_2_1000.png
│   │   │   ├── progression_image_2_1500.png
│   │   │   ├── progression_image_2_500.png
│   │   │   ├── progression_image_3_1000.png
│   │   │   ├── progression_image_3_1500.png
│   │   │   ├── progression_image_3_500.png
│   │   │   ├── progression_image_4_1000.png
│   │   │   ├── progression_image_4_1500.png
│   │   │   ├── progression_image_4_500.png
│   │   │   ├── progression_image_5_1000.png
│   │   │   ├── progression_image_5_1500.png
│   │   │   ├── progression_image_5_500.png
│   │   │   ├── progression_image_6_1000.png
│   │   │   ├── progression_image_6_1500.png
│   │   │   ├── progression_image_6_500.png
│   │   │   ├── progression_image_7_1000.png
│   │   │   ├── progression_image_7_1500.png
│   │   │   ├── progression_image_7_500.png
│   │   │   ├── progression_image_8_1000.png
│   │   │   ├── progression_image_8_1500.png
│   │   │   ├── progression_image_8_500.png
│   │   │   └── progression_image_9_500.png
│   │   ├── smooth_transition.png
│   │   └── woman_blonde.png
│   └── cifar10
│       ├── confusion_matrix_learned_features.png
│       ├── confusion_matrix_raw_pixels.png
│       ├── discriminator.png
│       ├── generated_images.png
│       ├── generator.png
│       ├── history
│       ├── progression_images
│       │   ├── progression_image_0_400.png
│       │   ├── progression_image_10_400.png
│       │   ├── progression_image_11_400.png
│       │   ├── progression_image_12_400.png
│       │   ├── progression_image_13_400.png
│       │   ├── progression_image_14_400.png
│       │   ├── progression_image_15_400.png
│       │   ├── progression_image_16_400.png
│       │   ├── progression_image_17_400.png
│       │   ├── progression_image_18_400.png
│       │   ├── progression_image_19_400.png
│       │   ├── progression_image_1_400.png
│       │   ├── progression_image_20_400.png
│       │   ├── progression_image_21_400.png
│       │   ├── progression_image_22_400.png
│       │   ├── progression_image_23_400.png
│       │   ├── progression_image_24_400.png
│       │   ├── progression_image_25_400.png
│       │   ├── progression_image_26_400.png
│       │   ├── progression_image_27_400.png
│       │   ├── progression_image_28_400.png
│       │   ├── progression_image_29_400.png
│       │   ├── progression_image_2_400.png
│       │   ├── progression_image_30_400.png
│       │   ├── progression_image_31_400.png
│       │   ├── progression_image_32_400.png
│       │   ├── progression_image_33_400.png
│       │   ├── progression_image_34_400.png
│       │   ├── progression_image_35_400.png
│       │   ├── progression_image_36_400.png
│       │   ├── progression_image_37_400.png
│       │   ├── progression_image_38_400.png
│       │   ├── progression_image_39_400.png
│       │   ├── progression_image_3_400.png
│       │   ├── progression_image_40_400.png
│       │   ├── progression_image_41_400.png
│       │   ├── progression_image_42_400.png
│       │   ├── progression_image_43_400.png
│       │   ├── progression_image_44_400.png
│       │   ├── progression_image_45_400.png
│       │   ├── progression_image_46_400.png
│       │   ├── progression_image_47_400.png
│       │   ├── progression_image_48_400.png
│       │   ├── progression_image_49_400.png
│       │   ├── progression_image_4_400.png
│       │   ├── progression_image_50_400.png
│       │   ├── progression_image_51_400.png
│       │   ├── progression_image_52_400.png
│       │   ├── progression_image_53_400.png
│       │   ├── progression_image_54_400.png
│       │   ├── progression_image_55_400.png
│       │   ├── progression_image_56_400.png
│       │   ├── progression_image_57_400.png
│       │   ├── progression_image_58_400.png
│       │   ├── progression_image_59_400.png
│       │   ├── progression_image_5_400.png
│       │   ├── progression_image_6_400.png
│       │   ├── progression_image_7_400.png
│       │   ├── progression_image_8_400.png
│       │   └── progression_image_9_400.png
│       └── sample_images.png
└── writeup
    ├── E4040.2021Spring.NBVV.report.nb2838.pdf
    ├── E4040.2021Spring.NBVV.report.nb2838.tex

12 directories, 197 files
```