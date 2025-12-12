export LMUData=/mnt/aigc/wangyubo/data/UG/data/benchmark/opensource_tsv
export HF_HOME=/mnt/aigc/shared_env/huggingface

cd /mnt/aigc/users/wangyubo/code/VLMEvalKit-worktree

# export VSI_PROMPT_TYPE='ruisi_prompt'
# export InternVL_ckpt_path='/mnt/umm/users/guchenyang/Repos/verl-internvl/spatial/checkpoints/base/internvl3.0/1019_egoexo*2_scannet*2_messytable_cot__sizemaze_scannetnotdeterminded_vsireldirv5_tf2023mcq_mindcuberot_sitemaze_vsr_spec_grounding_vica_1e-5/checkpoint-7000'

# export VSI_PROMPT_TYPE='aug'
# export InternVL_ckpt_path='/mnt/umm/users/guchenyang/Repos/verl-internvl/spatial/checkpoints/base/internvl3.0/1106_vsi_rel_dir_mindcube_COT_5e-6/checkpoint-700'
  # path rule sam as when we do cvpr


# export peter_test='abc'

# conda activate /mnt/aigc/yanglei/.conda/envs/vlm_eval_kit/

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun --master_port=29511 --nproc-per-node=4 /mnt/aigc/users/wangyubo/code/VLMEvalKit/run.py \
#   --data MUIRBench RoboSpatialHome RefSpatial_wo_unseen \
#   --model NEOov-2B-si-data32 \
#   --verbose --reuse \
#   --judge gpt-4o-1120 \
#   --work-dir /mnt/aigc/wangyubo/data/UG/data/resutls/jan_test/0125_test


CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc-per-node=4 /mnt/aigc/users/wangyubo/code/VLMEvalKit-worktree/run.py \
  --data CosmosReason1_4fps \
  --model Qwen2.5-VL-7B-Instruct \
  --verbose --reuse \
  --judge exact_matching \
  --work-dir /mnt/aigc/wangyubo/data/UG/data/resutls/feb_test/0213_test/cosmos/qwen_tp_0.6
