DEVICE=cuda:0
CFG=BMIP_ctx2_dp9_lr2

for SEED in 1
do
  bash scripts/bmip/base2new_train_bmip.sh eurosat ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh eurosat ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh imagenet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh imagenet ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh dtd ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh  dtd ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh ucf101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh  ucf101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh oxford_pets ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh  oxford_pets ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh food101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh  food101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh oxford_flowers ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh  oxford_flowers ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh sun397 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh  sun397 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh stanford_cars ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh  stanford_cars ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_train_bmip.sh caltech101 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bmip/base2new_test_bmip.sh  caltech101 ${SEED} ${DEVICE} ${CFG}
done