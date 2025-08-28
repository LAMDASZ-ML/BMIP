CFG=vit_b16_c2_ep10_ctx2_dp9TT_warmup5_BZ4_ex
DEVICE=cuda:3
load=15
for SEED in 1 2 3
do
    bash scripts/bitp/xd_test_best.sh food101 ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh caltech101 ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh oxford_flowers ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh oxford_pets ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh stanford_cars ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh sun397 ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh eurosat ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh ucf101 ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh dtd ${SEED} ${DEVICE} ${CFG} ${load}
    bash scripts/bitp/xd_test_best.sh imagenet ${SEED} ${DEVICE} ${CFG} ${load}
done