# KappaFace: Adaptive Additive Angular Margin Loss for Deep Face Recognition
Anonymous, "KappaFace: Adaptive Additive Angular Margin Loss for Deep Face Recognition"  <br /> 


* The core part of the code is based on <a href="https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch" target="_blank">Insightface</a>.

## Overview of our framework
<p align="center">
<!--     <img src="https://i.ibb.co/GddyQwv/diagram.png" width="960" alt="overall pipeline"> -->
    <img src="https://i.ibb.co/cNZpL0L/diagram-2.png"  width="780" alt="overall pipeline" alt="diagram-2" border="0">
<p>

## Datasets 
Training dataset:
<!-- - MSCeleb-1M: 5.8M images og 85k identities. -->
- CASIA-WebFace: 0.5M images  of  10k identities.
    
Test dataset:
- Labeled Faces in the Wild (LFW): 13k images of 5,749 identities.
- Cross-Age LFW (CALFW).
- Cross-Pose (CPLFW).
- YouTube Faces (YTF): 3,425 videos of 1,595 identities.
- Celebrities in Frontal-Prole (CFP).
- AgeDB-30: 12,240 images of 440 identities.
- IARPA Janus Benchmark: IJB-B and IJB-C.
- MegaFace: 1M images of 690k identities.
 <br /> 
We obtained the datasets from <a href="https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_" target="_blank">here</a>.

## Training
```
$ CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/base
```
    
## Results


**Table 1**  Verification results (%) on LFW, YTF, two pose benchmarks (CFP-FP and CPLFW) and two age benchmarks(AgeDB and CALFW).
    
|      Method      |  LFW  | CFP-FP | CPLFW | AgeDB | CALFW |  YTF |
|:----------------:|:-----:|:------:|:-----:|:-----:|:-----:|:----:|
|    Center Loss   | 99.27 |    -   | 81.40 |   -   | 90.30 | 94.9 |
|    SphereFace    | 99.27 |    -   | 81.40 |   -   | 90.30 | 95.0 |
|     VGGFace2     | 76.74 |    -   | 84.00 |   -   | 90.57 |   -  |
|      UV-GAN      | 99.60 |  94.05 |   -   |   -   |   -   |   -  |
|      ArcFace     | 99.82 |  98.27 | 92.08 | 98.15 | 95.45 | 98.0 |
|  CirricularFace  | 99.80 |  98.37 | 93.13 | 98.32 | 96.20 |   -  |
|    ArcFace-SCF   | 99.82 |  98.40 | 93.16 | 98.30 | 96.12 |   -  |
| KappaFace (_memory buffer_) | **99.83** |  **98.69** | **93.22** | **98.47** | **96.23** | **98.0** |
| KappaFace (_momentum encoder_) | **99.83** |  **98.60** | **93.40** | **98.35** | 96.15 | **98.0** |
    


**Table 2** 1:1 verification TAR (@FAR=$1e-4$) on the IJB-B and IJB-C datasets.
    
| Method                       | IJB-B | IJB-C |
|------------------------------|-------|-------|
| ResNet50+SENet50             | 80.0  | 84.1  |
| Multicolumn                  | 83.1  | 86.2  |
| P2SGrad                      | -     | 92.3  |
| Adacos                       | -     | 92.4  |
| ArcFace-VGG-R50              | 89.8  | 92.1  |
| ArcFace-MS1MV2-R100          | 94.2  | 95.6  |
| CurricularFace-MS1MV2-R100   | 94.8  | 96.1  |
| KappaFace-MS1MV2-R100 (_memory buffer_) | **95.1**  | **96.4**  |
| KappaFace-MS1MV2-R100 (_momentum encoder_) | **95.3**  | **96.6**  |
     
    
**Table 3** Verification comparison with SOTA methods on MegaFace Challenge 1 using FaceScrub as the probe set.
    
| Method                       |   Id  |  Ver  |
|------------------------------|:-----:|:-----:|
| AdaptiveFace                 | 95.02 | 95.61 |
| P2SGrad                      | 97.25 |   -   |
| Adacos                       | 97.41 |   -   |
| CosFace                      | 97.91 | 97.91 |
| MV-AM-Softmax-a              | 98.00 | 98.31 |
| ArcFace-MS1MV2-R100          | 98.35 | 98.48 |
| CurricularFace-MS1MV2-R100   | 98.71 | 98.64 |
| KappaFace-MS1MV2-R100 (_memory buffer_) | **98.77** | **98.91** |
| KappaFace-MS1MV2-R100 (_momentum encoder_) | **98.78** | **98.83** |
        

