from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.contrib import framework as contrib_framework

ORIGIN_DIR = "/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/m3l_b4_range1_origin_test/train"

if __name__ == '__main__':
    print_tensors_in_checkpoint_file(ORIGIN_DIR+"/model.ckpt-250000",
                                     tensor_name=None, all_tensors=False, all_tensor_names=True)