nohup python -u main.py --train --base configs/stableSRNew/v2-finetune_text_T_512.yaml --gpus 1,3,4,5, --name demo --scale_lr False &


WANDB_MODE=offline python -u main.py --train --base configs/stableSRNew/v2-finetune_text_T_512.yaml --gpus 4,5 --name demo --scale_lr False

CUDA_VISIBLE_DEVICES=1 proxychains python sr_val_ddpm_text_T_vqganfin_old.py --config configs/stableSRNew/v2-finetune_text_T_512.yaml --ckpt /NEW_EDS/JJ_Group/xutd/StableSR/logs/2024-07-12T14-17-20_demo/checkpoints/last.ckpt --init-img ./example --outdir ./out --ddpm_steps 1000 --dec_w 0.0 --seed 42 --n_samples 1 --vqgan_ckpt /NEW_EDS/JJ_Group/xutd/StableSR/512-base-ema.ckpt --colorfix_type nofix