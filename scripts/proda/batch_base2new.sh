for SEED in 1 2 3
do
#  bash scripts/proda/base2new_train_proda.sh fgvc_aircraft ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh fgvc_aircraft ${SEED} cuda:0
#  bash scripts/proda/base2new_train_proda.sh eurosat ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh eurosat ${SEED} cuda:0
  bash scripts/proda/base2new_train_proda.sh imagenet ${SEED} cuda:1
  bash scripts/proda/base2new_test_proda.sh imagenet ${SEED} cuda:1
#  bash scripts/proda/base2new_train_proda.sh dtd ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh  dtd ${SEED} cuda:0
#  bash scripts/proda/base2new_train_proda.sh ucf101 ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh  ucf101 ${SEED} cuda:0
#  bash scripts/proda/base2new_train_proda.sh oxford_pets ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh  oxford_pets ${SEED} cuda:0
#  bash scripts/proda/base2new_train_proda.sh food101 ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh  food101 ${SEED} cuda:0
#  bash scripts/proda/base2new_train_proda.sh oxford_flowers ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh  oxford_flowers ${SEED} cuda:0
#  bash scripts/proda/base2new_train_proda.sh sun397 ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh  sun397 ${SEED} cuda:0
#  bash scripts/proda/base2new_train_proda.sh stanford_cars ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh  stanford_cars ${SEED} cuda:0
#  bash scripts/proda/base2new_train_proda.sh caltech101 ${SEED} cuda:0
#  bash scripts/proda/base2new_test_proda.sh  caltech101 ${SEED} cuda:0
done