############## Configuration section begins ##################

# Model Config: [vitb32_CLIP, vitb16_CLIP, mae_vitb16, mocov3_vitb16, vit_base_patch16_224, vit_base_patch32_224, deit_base_patch16_224]
model_cfg=vitb14_CLIP # ViT-L/14@336px

# model_cfg=vitb16_CLIP
# For ViT-B/16 CLIP, you need to chance the input size.

# Mode: [proda]
mode=proda

# Use FP32 [default: True]
use_fp32=False

# Dataset: [caltech101]
dataset=$1
data_dir=$4

# Model checkpoint
model_ckpt=.
model_path=$5

# output directory
output_dir=$3

############ Configurations for hyperparameter tuning begin ############
# set to True to disable the automatic hyperparameter tuning
# and set the learning rate and weight accordingly below
# This option is only effective for linear probe and finetuning.

disable_hyperparameter_tuning=False

############ Configurations for hyperparameter tuning end   ############

############ Configurations for linear_probe/finetune begin ############

# Random seed: [0,1,2]
random_seed=$2

# Shots: {5, 20, 50} for few shot, and -1 for full-shot
num_shots=5

############ Configurations for linear_probe/finetune end   ############

############ Configurations for adding knowledge begin ############
# Please change the knowledge source accordingly.

############ Configurations for adding knowledge end   ############

############## Configuration section ends ##################


# Launching the job......

cd $6/vision_benchmark

python commands/proda.py --ds resources/datasets/$dataset.yaml --model resources/model/$model_cfg.yaml DATASET.NUM_SAMPLES_PER_CLASS $num_shots MODEL.CLIP_FP32 $use_fp32 DATASET.ROOT $data_dir OUTPUT_DIR $output_dir/$model_cfg/log TEST.MODEL_FILE $model_ckpt MODEL.CLIP_PATH $model_path DATASET.RANDOM_SEED_SAMPLING $random_seed