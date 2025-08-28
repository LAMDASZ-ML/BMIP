CFG=vit_b16_c2_ep10_ctx2_dp9TT_warmup5_BZ4_ex
DEVICE=cuda:3
load=15
for SEED in 1 2 3
do
    bash scripts/bitp/xd_test_best.sh imagenetv2 ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh imagenet_sketch ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh imagenet_a ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh imagenet_r ${SEED} ${DEVICE} ${CFG} ${load}
done