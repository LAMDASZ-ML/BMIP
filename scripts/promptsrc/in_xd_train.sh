CFG=$1
DEVICE=$2
# seed=1
bash scripts/promptsrc/xd_train.sh imagenet 1 ${DEVICE} ${CFG}
# seed=2
bash scripts/promptsrc/xd_train.sh imagenet 2 ${DEVICE} ${CFG}
# seed=3
bash scripts/promptsrc/xd_train.sh imagenet 3 ${DEVICE} ${CFG}
