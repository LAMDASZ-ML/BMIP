CFG=$1
Device=$2
load_epoch=$3
for SEED in 1 2 3
do
    bash scripts/promptsrc/xd_test.sh food101 ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh caltech101 ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh oxford_flowers ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh oxford_pets ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh stanford_cars ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh fgvc_aircraft ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh sun397 ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh eurosat ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh ucf101 ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh dtd ${SEED} ${Device} ${CFG} ${load_epoch}
    bash scripts/promptsrc/xd_test.sh imagenet ${SEED} ${Device} ${CFG} ${load_epoch}
done