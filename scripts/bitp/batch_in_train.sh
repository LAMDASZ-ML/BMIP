CFG=$2
DEVICE=$1
# seed=1
bash scripts/bitp/xd_train.sh imagenet 1 ${DEVICE} ${CFG}
# seed=2
bash scripts/bitp/xd_train.sh imagenet 2 ${DEVICE} ${CFG}
# seed=3
bash scripts/bitp/xd_train.sh imagenet 3 ${DEVICE} ${CFG}