for SEED in 1 2 3
do
  bash scripts/vpt/base2new_train_vpt.sh fgvc_aircraft ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh fgvc_aircraft ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh eurosat ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh eurosat ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh imagenet ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh imagenet ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh dtd ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh  dtd ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh ucf101 ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh  ucf101 ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh oxford_pets ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh  oxford_pets ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh food101 ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh  food101 ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh oxford_flowers ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh  oxford_flowers ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh sun397 ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh  sun397 ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh stanford_cars ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh  stanford_cars ${SEED} cuda:0
  bash scripts/vpt/base2new_train_vpt.sh caltech101 ${SEED} cuda:0
  bash scripts/vpt/base2new_test_vpt.sh  caltech101 ${SEED} cuda:0
done