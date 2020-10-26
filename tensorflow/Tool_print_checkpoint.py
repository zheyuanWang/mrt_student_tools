from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.contrib import framework as contrib_framework

ORIGIN_DIR = "/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/m3l_b4_r1_origin_350k_test"
TMP_DIR = "/mrtstorage/users/zwang/github_zheyuan/mrt_experiments/tests/m3l_b1_range1_n8_ck0_f2_reusetest4/train/model.ckpt-0"
PARA_8_DIR = "/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/m3l_b4_range1_ckALL_parallel8/train"

if __name__ == '__main__':
    print_tensors_in_checkpoint_file(ORIGIN_DIR+"/model.ckpt-250000",tensor_name=None, all_tensors=False, all_tensor_names=True)
    #print_tensors_in_checkpoint_file(TMP_DIR,tensor_name=None, all_tensors=False, all_tensor_names=True)