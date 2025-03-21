export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
python train_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --resolution=1024 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=4 \
  --max_train_steps=10 \
  --max_train_samples=12 \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="bf16" \
  --output_dir="sdxl-naruto-model"