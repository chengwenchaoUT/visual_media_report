DATA_DIR = './'

SMPL_PATH_MALE = './smpl/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
SMPL_PATH_FEMALE = './smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'

GAR_INFO_FILE = 'garment_class_info.pkl'

SMOOTH_STORED = True

POSE_SPLIT_FILE = 'split_static_pose_shape.npz'

# Lists the indices of joints which affect the deformations of particular garment
VALID_THETA = {
    't-shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19],
    'old-t-shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19],
    'shirt': [0, 1, 2, 3, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    'pants': [0, 1, 2, 4, 5, 7, 8],
    'skirt' : [0, 1, 2, ],
}