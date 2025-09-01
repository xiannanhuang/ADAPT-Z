# The implementation of ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space)
This code is the implementation of ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space) algorithm, along with supplementary experiments from the paper and implementations of baselines.

This codebase is primarily based on DSOF (Fast and Slow Streams for Online Time Series Forecasting Without Information Leakage, ICLR 2025; paper: https://openreview.net/pdf?id=I0n3EyogMi, repository: https://github.com/yyalau/iclr2025_dsof)
## 1) Train basemodel
run train_basemodel.py to obtain base model

## 2) Online deployment
run adapt-z.py to conduct online prediction

## Baselines and 
run run_dsof.py to conduct experiments using DSOF
run ADCSD.py to conduct experiments using ADCSD
run parameter_finetune to conduct experiments using OGD
run z_finetune to conduct experiments using f-OGD
run train_FAN.py/train_DishTS.py and ada_online2 fan.py/ada_online dishts.py to conduct experiments using FAN/DishTS and FAN+/DishYS+
