model_name=PRNet

root_path_name=./dataset/solar
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar

seq_len=96
random_seed=2024
# ==========================================
# 1. 预测长度 96
# ==========================================
pred_len=96
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 137 \
  --cycle 144 \
  --train_epochs 30 \
  --patience 5 \
  --use_revin 0 \
  --d_model 512 \
  --reduction 16 \
  --itr 1 --batch_size 32 --learning_rate 0.0007 --random_seed $random_seed
  

# ==========================================
# 2. 预测长度 192
# ==========================================
pred_len=192
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 137 \
  --cycle 144 \
  --train_epochs 30 \
  --patience 5 \
  --use_revin 0 \
  --d_model 512 \
  --reduction 16 \
  --itr 1 --batch_size 32 --learning_rate 0.001 --random_seed $random_seed
# ==========================================
# 3. 预测长度 336
# ==========================================
pred_len=336
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 137 \
  --cycle 144 \
  --train_epochs 30 \
  --patience 5 \
  --use_revin 0 \
  --d_model 512 \
  --reduction 16 \
  --itr 1 --batch_size 32 --learning_rate 0.001 --random_seed $random_seed

# ==========================================
# 4. 预测长度 720
# ==========================================
pred_len=720
python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 137 \
  --cycle 144 \
  --train_epochs 30 \
  --patience 5 \
  --use_revin 0 \
  --d_model 512 \
  --reduction 16 \
  --itr 1 --batch_size 32 --learning_rate 0.0015 --random_seed $random_seed
