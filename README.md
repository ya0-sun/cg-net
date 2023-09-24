# CG-Net
This repo contains code of the paper: [CG-Net: Conditional GIS-Aware network for individual building segmentation in VHR SAR images](https://ieeexplore.ieee.org/document/9321533)

## Intro
This article addresses the issue of individual building segmentation from a single VHR SAR image in large-scale urban areas. We introduce building footprints from geographic information system (GIS) data as complementary information and propose a novel conditional GIS-aware network (CG-Net). Additionally, we propose an approach of ground truth generation of buildings from an accurate digital elevation model (DEM), which can be used to generate large-scale SAR image data sets. 

The method is validated using a high-resolution spotlight TerraSAR-X image collected over Berlin. Experimental results show that the proposed CG-Net effectively brings improvements with variant backbones. The segmentation results can be applied to reconstruct LoD1 3D building models, which is demonstrated in our experiments.

![berlinHS_result](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/36/9633014/9321533/zhu17-3043089-large.gif)

*Segmentation results in the study area obtained by DeepLabv3-CG. The building segments are plotted with different colors translucently for visualizing the layover areas between buildings. rg and az denote the range direction and the azimuth direction, respectively.*

![lod1](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/36/9633014/9321533/zhu18-3043089-large.gif)

*LoD1 building models reconstructed from the segmentation results in the study area, superimposed on the SAR image. The absolute mean error of estimated building heights in the study site is 2.39 m. Layover areas of some buildings are visible, as pointed by the yellow and red arrows. Building heights are color-coded.*

## Code 
The CG module is integrated with FCN and DeepLabV3 in the code.

FCN-CG: 

* Train: train_fcn.py 
* Test: test_fcn.py

DeepLabV3-CG: 

* Train: train_deeplabv3.py
* Test: test_deeplabv3.py
## Citation

If you find the repo useful, please cite the following paper:

```
@ARTICLE{SUN2020cgnet,
  author={Sun, Yao and Hua, Yuansheng and Mou, Lichao and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={CG-Net: Conditional GIS-Aware Network for Individual Building Segmentation in VHR SAR Images}, 
  year={2022},
  volume={60},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2020.3043089}
}
```
