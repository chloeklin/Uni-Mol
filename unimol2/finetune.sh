



task_name="qm9dft_v2"  # molecular property prediction task name 
task_num=3
weight_name="checkpoint.pt"
loss_func="finetune_smooth_mae"
arch_name="1.1B"
arch=unimol2_$arch_name


data_path="./unimol2/data/"
weight_path="./"
weight_path=$weight_path/$weight_name

drop_feat_prob=1.0
use_2d_pos=0.0
ema_decay=0.999

lr=1e-4
batch_size=32
epoch=40
dropout=0
warmup=0.06
local_batch_size=16
seed=0
conf_size=11

n_gpu=1
reg_task="--reg"
metric="valid_agg_mae"
save_dir="./save_dir"

update_freq=`expr $batch_size / $local_batch_size`
global_batch_size=`expr $local_batch_size \* $n_gpu \* $update_freq`

torchrun --standalone --nnodes=1 --nproc_per_node=$n_gpu \
    $(which unicore-train) $data_path \
    --task-name $task_name --user-dir ./unimol2 --train-subset train --valid-subset valid,test \
    --conf-size $conf_size \
    --num-workers 8 --ddp-backend=c10d \
    --task mol_finetune --loss $loss_func --arch $arch  \
    --classification-head-name $task_name --num-classes $task_num \
    --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
    --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch \
    --batch-size $local_batch_size --pooler-dropout $dropout\
    --update-freq $update_freq --seed $seed \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --no-save \
    --log-interval 100 --log-format simple \
    --validate-interval 1 \
    --finetune-from-model $weight_path \
    --best-checkpoint-metric $metric --patience 20 \
    --save-dir $save_dir \
    --drop-feat-prob ${drop_feat_prob} \
    --use-2d-pos-prob ${use_2d_pos} \
    $more_args \
    $reg_task \
    --find-unused-parameters