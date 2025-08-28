for SEED in 1 2 3
do
#  bash scripts/upt/base2new_train_upt.sh fgvc_aircraft ${SEED} cuda:0
  bash scripts/upt/base2new_test_upt.sh fgvc_aircraft ${SEED} cuda:0
#  bash scripts/upt/base2new_train_upt.sh eurosat ${SEED} cuda:0
  bash scripts/upt/base2new_test_upt.sh eurosat ${SEED} cuda:0
#  bash scripts/upt/base2new_train_upt.sh oxford_flowers ${SEED} cuda:1
  bash scripts/upt/base2new_test_upt.sh oxford_flowers ${SEED} cuda:1
#  bash scripts/upt/base2new_train_upt.sh imagenet ${SEED} cuda:1
  bash scripts/upt/base2new_test_upt.sh imagenet ${SEED} cuda:1
#  bash scripts/upt/base2new_train_upt.sh dtd ${SEED} cuda:1
  bash scripts/upt/base2new_test_upt.sh dtd ${SEED} cuda:1
#  bash scripts/upt/base2new_train_upt.sh ucf101 ${SEED} cuda:1
  bash scripts/upt/base2new_test_upt.sh ucf101 ${SEED} cuda:1
#  bash scripts/upt/base2new_train_upt.sh oxford_pets ${SEED} cuda:1
  bash scripts/upt/base2new_test_upt.sh oxford_pets ${SEED} cuda:1
#  bash scripts/upt/base2new_train_upt.sh food101 ${SEED} cuda:1
  bash scripts/upt/base2new_test_upt.sh food101 ${SEED} cuda:1
#  bash scripts/upt/base2new_train_upt.sh sun397 ${SEED} cuda:1
  bash scripts/upt/base2new_test_upt.sh sun397 ${SEED} cuda:1
#  bash scripts/upt/base2new_train_upt.sh stanford_cars ${SEED} cuda:1
  bash scripts/upt/base2new_test_upt.sh stanford_cars ${SEED} cuda:1
#  bash scripts/upt/base2new_train_upt.sh caltech101 ${SEED} cuda:1
  bash scripts/upt/base2new_test_upt.sh caltech101 ${SEED} cuda:1
done