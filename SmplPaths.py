import pickle as pkl
from local_config import SMPL_PATH_FEMALE, SMPL_PATH_MALE
from smpl_lib.serialization import backwards_compatibility_replacements, load_model
import numpy as np
import scipy.sparse as sp

def get_hrmesh(v, f):
    """
    get a high resolution version of the mesh
    :param v: vertices
    :param f: faces
    :return:
        nv: new vertices
        nf: new faces
        mapping: mapping from low res to high
    """
    from geometry import loop_subdivider
    (mapping ,nf) = loop_subdivider(v, f)
    nv = mapping.dot(v.ravel()).reshape(-1, 3)
    return (nv, nf, mapping)

class SmplPaths:
    def __init__(self, gender):
        self.gender = gender

    def get_smpl_file_path(self):
        if self.gender == 'male':
            return SMPL_PATH_MALE
        elif self.gender == 'female':
            return SMPL_PATH_FEMALE
        else:
            print("no such gender file")

    def get_smpl_data(self):
        dd = pkl.load(open(self.get_smpl_file_path(), 'rb'), encoding='latin1')
        backwards_compatibility_replacements(dd)

        hv, hf, mapping = get_hrmesh(dd['v_template'], dd['f'])

        num_betas = dd['shapedirs'].shape[-1]
        J_reg = dd['J_regressor'].asformat('csr')

        model = {
            'v_template': hv,
            'weights': np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(dd['weights'][:, i], -1), 3)).reshape(-1, 3)
                        , axis=1),
                    axis=-1)
                for i in range(24)
            ]),
            'posedirs': mapping.dot(dd['posedirs'].reshape((-1, 207))).reshape(-1, 3, 207),
            'shapedirs': mapping.dot(dd['shapedirs'].reshape((-1, num_betas))).reshape(-1, 3, num_betas),
            'J_regressor': sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr), shape=(24, hv.shape[0])),
            'kintree_table': dd['kintree_table'],
            'bs_type': dd['bs_type'],
            'bs_style': dd['bs_style'],
            'J': dd['J'],
            'f': hf,
        }

        return model
