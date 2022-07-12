# This is an example of how to set your own training script
# For more parameters settings, please refer to the get_config.py
olf_root="Data"
experiment_name="Debug"
cuda_idx="7"
task="olf-seg-only"
model="DeepLabV3_ResNet101"
batch_size=6
num_epochs=10
num_epochs_decay=10
num_workers=16
eval_frequency=5
lr=0.0002
lr_Scheduler='ExponentialLR'
loss_type='SSLoss'
aug_prob=0.5

python main.py --cuda_idx $cuda_idx \
               --seed   10    \
               --task   $task \
               --olf_root $olf_root \
               --experiment_name $experiment_name \
               --num_epochs $num_epochs \
               --num_workers $num_workers \
               --batch_size $batch_size \
               --num_epochs_decay $num_epochs_decay \
               --eval_frequency $eval_frequency \
               --lr $lr \
               --lr_Scheduler $lr_Scheduler \
               --loss_type $loss_type \
               --aug_prob $aug_prob
