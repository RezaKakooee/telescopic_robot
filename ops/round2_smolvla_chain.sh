set -e
cd /home/azureuser/telescopic_robot
export MUJOCO_GL=egl PYTHONPATH=.
D1=storage_local/20260914_1612__local__generate_inspection_demos__30ep
D2=storage_local/20260914_2331__local__generate_inspection_demos__30ep
echo "=== warehouse demos"
/home/azureuser/miniconda3/envs/roboverse/bin/python scripts/data/generate_inspection_demos.py --episodes 30 --workers 1 --seed-offset 2000 --courses inspection_warehouse --out $D2 2>&1 | grep -E "kept|done in"
OUT=storage_local/$(date +%Y%m%d_%H%M)__local__lerobot_dataset__inspection_tours_x2_os4
mkdir -p $OUT
echo "=== convert -> $OUT"
/home/storage_group/envs/lerobot/bin/python scripts/data/convert_inspection_demos_to_lerobot.py --demos $D1 $D2 --root $OUT/lerobot --oversample 4 2>&1 | grep -E "^wrote|Error|Traceback"
TR=storage_local/$(date +%Y%m%d_%H%M)__local__train_smolvla__inspection_tours_x2_os4
mkdir -p $TR
echo "=== train -> $TR"
/home/storage_group/envs/lerobot/bin/lerobot-train --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda --dataset.repo_id=roboball/inspection_tours --dataset.root=$PWD/$OUT/lerobot --rename_map='{"observation.image": "observation.images.camera1"}' --batch_size=32 --steps=20000 --log_freq=100 --save_freq=5000 --output_dir=$TR/train --job_name=smolvla_round2 --wandb.enable=false > $TR/train.log 2>&1
echo "=== eval"
C=$TR/train/checkpoints/020000/pretrained_model
/home/storage_group/envs/lerobot/bin/python scripts/vla/eval_smolvla_inspection.py --checkpoint $C --replan 1 --video 2>&1 | grep -E "^inspection|reached"
echo "=== ROUND2 DONE"
