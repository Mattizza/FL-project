# Towards Real World Federated Learning
### Machine Learning and Deep Learning 2023
#### Politecnico di Torino
Code for the Federated Learning project. Checkpoints of the performed experiments can be found at this [link](https://drive.google.com/drive/folders/1ZAe2BeIY9TzB0Y22fVszJbyksgl-RDLE?usp=sharing).

## Setup
#### Environment
You can use CoLab to run the code.

#### Datasets
The repository supports experiments on the following datasets:
1. Reduced **Federated IDDA** from FedDrive [1]
   - Task: semantic segmentation for autonomous driving
   - 24 users
2. Reduced **GTA 5** datasets [2]

## How to run
The ```main.py``` orchestrates training. All arguments need to be specified through the ```args``` parameter (options can be found in ```utils/args.py```).

CoLab notebook examples can be found in the homonymous folder.

## References
[1] Fantauzzo, Lidia, et al. "FedDrive: generalizing federated learning to semantic segmentation in autonomous driving." 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022.

[2] Richter, S.R., Vineet, V., Roth, S., Koltun, V. (2016). Playing for Data: Ground Truth from Computer Games. In: Leibe, B., Matas, J., Sebe, N., Welling, M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science(), vol 9906. Springer, Cham. https://doi.org/10.1007/978-3-319-46475-6_7
