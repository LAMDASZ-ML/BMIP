for SEED in 7 8 9
do
  bash scripts/maple/base2new_train_maple1.sh fgvc_aircraft ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh fgvc_aircraft ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh eurosat ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh eurosat ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh imagenet ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh imagenet ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh dtd ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh dtd ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh ucf101 ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh ucf101 ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh oxford_pets ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh oxford_pets ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh food101 ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh food101 ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh oxford_flowers ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh oxford_flowers ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh sun397 ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh sun397 ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh stanford_cars ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh stanford_cars ${SEED} cuda:0
  bash scripts/maple/base2new_train_maple1.sh caltech101 ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple1.sh caltech101 ${SEED} cuda:0
done