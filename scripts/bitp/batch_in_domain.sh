CFG=$1
load_epoch=$2
for SEED in 1 2 3
do
   bash scripts/bitp/xd_test.sh imagenetv2 ${SEED} cuda:1 ${CFG} ${load_epoch}
   bash scripts/bitp/xd_test.sh imagenet_sketch ${SEED} cuda:1 ${CFG} ${load_epoch}
   bash scripts/bitp/xd_test.sh imagenet_a ${SEED} cuda:1 ${CFG} ${load_epoch}
   bash scripts/bitp/xd_test.sh imagenet_r ${SEED} cuda:1 ${CFG} ${load_epoch}
done