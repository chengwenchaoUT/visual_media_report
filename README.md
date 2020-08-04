# visual_media_report

visual media report

the paper I selected:  "TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style", CVPR2020 (https://arxiv.org/pdf/2003.04583.pdf)

1. Why this paper is important?
This paper presents TailorNet, a neural model which predicts deformation in 3D as a function of three factors: pose, shape and style. Previous models either model deformation due to pose for a fixed shape, shape and pose for a fixed style, or style for a fixed pose. TailorNet models the effects of pose, shape and style jointly. Futhermore, existing joint models for pose and shape often produce over-smooth results(even for a fixed style). TailorNet's hypothesis is that combinations  of examples smooth out high frequency components such as fine-wrinkles, which makes learning the three factors jointly hard. At the heart of TailorNet is a decomposition of deformation into a high-frequency and a low frequency component.
The low-frequency component is predicted from pose, shape and style parameters with an MLP network. The high-frequency component is predicted with a mixture of shape-style specific pose models.The weights of the mixture are computed with a narrow bandwidth kernel to guarantee that only predictions with similar high-frequency patterns are combined. The style variation is obtained by computing, in a canonical pose, a subspace of deformation, which satisfies physical constraints such as inter-penetration, and draping on the body.

![image](https://github.com/chengwenchaoUT/visual_media_report/blob/master/imgs/TailorNet.png)

