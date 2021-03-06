# visual_media_report

visual media report  

the paper I selected:  "TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style", CVPR2020 (https://arxiv.org/pdf/2003.04583.pdf)  

__1. Why this paper is important?__    
This paper presents TailorNet, a neural model which predicts deformation in 3D as a function of three factors: pose, shape and style. Previous models either model deformation due to pose for a fixed shape, shape and pose for a fixed style, or style for a fixed pose. TailorNet models the effects of pose, shape and style jointly. Futhermore, existing joint models for pose and shape often produce over-smooth results(even for a fixed style). TailorNet's hypothesis is that combinations  of examples smooth out high frequency components such as fine-wrinkles, which makes learning the three factors jointly hard. At the heart of TailorNet is a decomposition of deformation into a high-frequency and a low frequency component.    
The low-frequency component is predicted from pose, shape and style parameters with an MLP network. The high-frequency component is predicted with a mixture of shape-style specific pose models.The weights of the mixture are computed with a narrow bandwidth kernel to guarantee that only predictions with similar high-frequency patterns are combined. The style variation is obtained by computing, in a canonical pose, a subspace of deformation, which satisfies physical constraints such as inter-penetration, and draping on the body.    


Following is an overview of TailorNet.(obtained from https://arxiv.org/pdf/2003.04583.pdf)
![image](https://github.com/chengwenchaoUT/visual_media_report/blob/master/imgs/TailorNet.png)  


1.1 garment model aligned with SMPL  
SMPL represents the human body M(·) as a parametric function of pose(θ) and shape(β):  
       &emsp;&emsp;&emsp;                 M(β, θ) = W(T(β, θ), J(β), θ,W)   
        &emsp;&emsp;&emsp;                T(β, θ) = T + Bs(β) + Bp(θ)  
        &emsp;&emsp;&emsp;          T: base mesh vertices T in a T-pose  
&emsp;&emsp;&emsp;              W(·): skining function  


For a given style D, shape β and pose θ, TailorNet deforms clothing using the un-posed SMPL function T(θ, β):  
&emsp;&emsp;&emsp;   T<sup>G</sup>(β, θ, D) = I T(β, θ) + D  
then the final cloth can be obtained by following formula:  
&emsp;&emsp;&emsp;   G(β, θ, D) = W(T<sup>G</sup>(β, θ, D), J(β), θ,W)  

1.2 Un-posing Garment Deformation  
Given a set of simulated garments G for a given pose θ and shape β, TailorNet disentangles non-rigid deformation
from articulation by un-posing:    
 &emsp;&emsp;&emsp;    D = W<sup>−1</sup>(G, J(β), θ,W) − I T(β, θ)  
  &emsp;&emsp;&emsp;where W<sup>−1</sup>() is inverse function of W()  
  
Non-rigid deformation D in the unposed space is affected by body pose, shape and the garment style (size, sleeve length, fit). Hence, TailorNet proposes to learn deformation D as a function of shape β, pose θ and style γ,
i.e. D(β, θ, γ) : R<sup>|θ|</sup> × R<sup>|β|</sup> × R<sup>|γ|</sup> → R<sup>m×3</sup>.   
  
  
__2.What I have implemented?__    
  
base_trainer.py  
class Trainer(object): Implements trainer class for TailorNet low frequency predictor  
            def __init__(self, params):  initializes trainer class from params  
            def load_data(self, split): return dataset and dataloader  
            def build_model(self): build MLP network for low frequency regressor  
            def train(self): training network  
            def onestep(self, inputs): one step during training  
              
 
hf_trainer.py  
class HFTrainer(base_trainer.Trainer): Implements trainer class for TailorNet high frequency predictor, overloads some functions of base_trainer.Trainer class  
          

cannon_trainer.py  
class CannonTrainer(base_trainer.Trainer): Implements trainer class to predict deformations in canonical pose, overloads some functions of base_trainer.Trainer class  


local_config.py  
DATA_DIR: dataset root directory  
SMPL_PATH_MALE: paths to SMPL male model   
SMPL_PATH_FEMALE: paths to SMPL female model   
GAR_INFO_FILE: paths to garment information file   
SMOOTH_STORED: Indicates that smooth groundtruth data is available or not.  
POSE_SPLIT_FILE: path to train/test splits file  
VALID_THETA: Lists the indices of joints which affect the deformations of particular garment  


networks.py  
class FullyConnected(nn.Module): fullly connected network class  


dataset.py  
def flip_theta(theta, batch=False): flip SMPL theta along y-z plane  
def get_Apose(): function of getting apose parameters  
class PivotsStyleShape(Dataset): dataset class for all style-shape datasets of pivots and the concate them  
class OneStyleShape(Dataset): dataset class for one style-shape datasets  
class OneStyleShapeHF(OneStyleShape): dataset class for one style-shape high frequency datasets  
class ShapeStyleInCannonPose(Dataset): dataset class for style-shape datasets in cannon poses  


SmplPath.py  
def get_hrmesh(v, f): get a high resolution version of given mesh  
class SmplPaths: get smpl data from given file path  


SMPLToGarment.py  
class SMPLToGarment(object): SMPL class for garments  
class TorchSMPLToGarment(nn.Module): SMPL class for garments, a torch version  
  
  
  
  
  
__3. How to train the model?__    
step 1: Register and download SMPL models(https://smpl.is.tue.mpg.de/en)  
step 2: Download dataset for TailorNet(https://github.com/zycliao/TailorNet_dataset)  
step 3: Run base_trainer.py, hf_trainer.py and cannon_trainer.py  
