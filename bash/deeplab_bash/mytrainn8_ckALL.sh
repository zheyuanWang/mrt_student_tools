# Exit immediately if a command exits with a non-zero status.
set -e
#readlink -f ${0}
#wzy: use the right visual enviroment
. /home/zwang/venv/bin/activate

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
# python "${WORK_DIR}"/model_test.py

#backbone_batchsize_datasetVersion_NrOfTest_NrOfParallelLayers_intervarBetweenFrames_ifLoadCheckpoint_NrOfFrames e.p. m3l_b1_range1_n3_int3_ck0_f3
EXP_DIR="/home/zwang/Documents/mrt_experiments/tests/m3l_b4_r1_n8_int1_ckALL_6"

  # if error = Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint
  # : change the model dir where you save your old checkpoints.

cd "${CURRENT_DIR}"

mkdir -p "${EXP_DIR}/train"
cp "${WORK_DIR}"/"${0}" "${EXP_DIR}"/.

CUDA_VISIBLE_DEVICES=0 python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="mobilenet_v3_large_seg" \
  --dataset="skitti_gm" \
  --train_crop_size="502,1002" \
  --image_pooling_stride=4,5 \
  --aspp_convs_filters=128 \
  --aspp_with_concat_projection=0 \
  --aspp_with_squeeze_and_excitation=1 \
  --decoder_use_sum_merge=1 \
  --decoder_filters=19 \
  --decoder_output_is_logits=0 \
  --image_se_uses_qsigmoid=1 \
  --image_pyramid=1 \
  --output_stride=8 \
  --decoder_output_stride=8 \
  --train_batch_size=4 \
  --training_number_of_steps=350000 \
  --if_training=1 \
    --log_steps=50 \
    --save_interval_secs=6000 \
    --save_summaries_secs=6000 \
--number_parallel_layers=8 \
--interval_between_frames=1 \
  --fine_tune_batch_norm=false \
  --train_logdir="${EXP_DIR}/train" \
  --dataset_dir="/mrtstorage/users/zwang/pcd_mapper_pastonly/polar_2frames/train" \
  --tf_initial_checkpoint="/mrtstorage/users/zwang/github_zheyuan/share_files/checkpoints/m3l_b4_range1_ckALL_parallel8/train/model.ckpt-250000"


