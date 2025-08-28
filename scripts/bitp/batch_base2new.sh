CFG=vit_b16_c2_ep15_ctx2_dp9TT_warmup5_BZ4_ex_seed11PT4
DEVICE=cuda:2
for SEED in 1
do
  # bash scripts/bitp/base2new_train_bitp.sh eurosat ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh eurosat ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_train_bitp.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh fgvc_aircraft ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_train_bitp.sh imagenet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh imagenet ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_train_bitp.sh dtd ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh  dtd ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_train_bitp.sh ucf101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh  ucf101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_train_bitp.sh oxford_pets ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh  oxford_pets ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_train_bitp.sh food101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh  food101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_train_bitp.sh oxford_flowers ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh  oxford_flowers ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_train_bitp.sh sun397 ${SEED} ${DEVICE} ${CFG}
  bash scripts/bitp/base2new_test_bitp.sh  sun397 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_train_bitp.sh stanford_cars ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh  stanford_cars ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_train_bitp.sh caltech101 ${SEED} ${DEVICE} ${CFG}
  # bash scripts/bitp/base2new_test_bitp.sh  caltech101 ${SEED} ${DEVICE} ${CFG}
done