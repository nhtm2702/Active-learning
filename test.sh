#function run
run() {
    number=$1
    shift
    for i in `seq $number`; do
      $@
    done
}

   CUDA_VISIBLE_DEVICES=0 /kaggle/working/CAMPAL/main.py \
    --model resnet18_cifar \
    --dataset cifar10 \
    --strategy MarginSampling \
    --num-init-labels 100 \
    --n-cycle 2 \
    --num-query 100 \
    --n-epoch 1 \
    --subset 50000 \
    --seed 43 \
    --batch-size 50 \
    --lr 0.1 \
    --momentum 0.9 \
    --weight-decay 0.0005 \
    --aug-lab-on \
    --aug-ratio-lab 10 \
    --mix-ratio-lab 5 \
    --aug-lab-training-mode StrengthGuidedAugment \
    --aug-lab-strength-mode all \
    --aug-ulb-on \
    --aug-ratio-ulb 1 \
    --mix-ratio-ulb 1 \
    --aug-ulb-evaluation-mode StrengthGuidedAugment \
    --aug-ulb-strength-mode all \
    --aug-metric-ulb min