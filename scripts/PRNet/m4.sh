#!/bin/bash

# M4 Dataset Forecasting Script for PRNet
# This script contains the configurations for all 6 subsets of the M4 dataset.
# The --cycle parameter is set based on the physical seasonality of the data,
# which allows PRNet's LatentChannelRecalibration and Temporal Query Prior to shine.

model_name=PRNet
root_path_name=./dataset/m4

# ---------------------------------------------------------
# 1. Yearly (年度数据)
# Physical Cycle: 1 (Long-term macro trend, no short-term seasonality)
# Horizon: 6
# ---------------------------------------------------------
python run.py \
  --random_seed 2026 \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --features M \
  --seasonal_patterns 'Yearly' \
  --cycle 1 \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 6 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --loss SMAPE \
  --des 'Exp' \
  --itr 1

# ---------------------------------------------------------
# 2. Quarterly (季度数据)
# Physical Cycle: 4 (4 quarters in a year)
# Horizon: 8
# ---------------------------------------------------------
python run.py \
  --random_seed 2026 \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id m4_Quarterly \
  --model $model_name \
  --data m4 \
  --features M \
  --seasonal_patterns 'Quarterly' \
  --cycle 4 \
  --seq_len 16 \
  --label_len 8 \
  --pred_len 8 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --loss SMAPE \
  --des 'Exp' \
  --itr 1

# ---------------------------------------------------------
# 3. Monthly (月度数据)
# Physical Cycle: 12 (12 months in a year)
# Horizon: 18
# ---------------------------------------------------------
python run.py \
  --random_seed 2026 \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --seasonal_patterns 'Monthly' \
  --cycle 12 \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 18 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --loss SMAPE \
  --des 'Exp' \
  --itr 1

# ---------------------------------------------------------
# 4. Weekly (周度数据)
# Physical Cycle: 52 (52 weeks in a year, capturing annual seasonality)
# Horizon: 13
# ---------------------------------------------------------
python run.py \
  --random_seed 2026 \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id m4_Weekly \
  --model $model_name \
  --data m4 \
  --features M \
  --seasonal_patterns 'Weekly' \
  --cycle 52 \
  --seq_len 26 \
  --label_len 13 \
  --pred_len 13 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --loss SMAPE \
  --des 'Exp' \
  --itr 1

# ---------------------------------------------------------
# 5. Daily (日度数据)
# Physical Cycle: 7 (7 days in a week, capturing weekend/weekday effects)
# Horizon: 14
# ---------------------------------------------------------
python run.py \
  --random_seed 2026 \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id m4_Daily \
  --model $model_name \
  --data m4 \
  --features M \
  --seasonal_patterns 'Daily' \
  --cycle 7 \
  --seq_len 28 \
  --label_len 14 \
  --pred_len 14 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --loss SMAPE \
  --des 'Exp' \
  --itr 1

# ---------------------------------------------------------
# 6. Hourly (小时数据)
# Physical Cycle: 24 (24 hours in a day)
# Horizon: 48
# ---------------------------------------------------------
python run.py \
  --random_seed 2026 \
  --is_training 1 \
  --root_path $root_path_name \
  --model_id m4_Hourly \
  --model $model_name \
  --data m4 \
  --features M \
  --seasonal_patterns 'Hourly' \
  --cycle 24 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 48 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --loss SMAPE \
  --des 'Exp' \
  --itr 1
