olf_root="Data"
experiment_name="U_Net"
cuda_idx="5"
task="olf-seg-only"
model="U_Net"
batch_size=4
num_epochs=300
num_epochs_decay=10
num_workers=16
eval_frequency=10
lr=0.0002
lr_Scheduler='ReduceLROnPlateau'

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
               --lr_Scheduler $lr_Scheduler