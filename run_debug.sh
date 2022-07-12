olf_root="Data"
experiment_name="U_Net_Test_Threshold"
cuda_idx="7"
task="olf-seg-only"
model="U_Net"
batch_size=6
num_epochs=100
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
