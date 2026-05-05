model_name=PRNet
root_path_name=./dataset/
data_path_name=PEMS04.npz
model_id_name=PEMS04
data_name=PEMS
seq_len=96
random_seed=2026

# ==========================================
# 1. 预测长度 12
# ==========================================
pred_len=12
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
  --enc_in 307 \
  --cycle 288 \
  --train_epochs 30 \
  --patience 5 \
  --use_revin 0 \
  --d_model 512 \
  --reduction 16 \
  --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed

# ==========================================
# 2. 预测长度 24
# ==========================================
pred_len=24
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
  --enc_in 307 \
  --cycle 288 \
  --train_epochs 30 \
  --patience 5 \
  --use_revin 0 \
  --d_model 512 \
  --reduction 16 \
  --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed

# ==========================================
# 3. 预测长度 48
# ==========================================
pred_len=48
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
  --enc_in 307 \
  --cycle 288 \
  --train_epochs 30 \
  --patience 5 \
  --use_revin 0 \
  --d_model 512 \
  --reduction 16 \
  --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed

# ==========================================
# 4. 预测长度 96
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
  --enc_in 307 \
  --cycle 288 \
  --train_epochs 30 \
  --patience 5 \
  --use_revin 0 \
  --d_model 512 \
  --reduction 16 \
  --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed