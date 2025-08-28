for SEED in 1 2 3
do
  bash scripts/maple/base2new_train_maple2.sh fgvc_aircraft ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh fgvc_aircraft ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh eurosat ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh eurosat ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh imagenet ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh imagenet ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh dtd ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh dtd ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh ucf101 ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh ucf101 ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh oxford_pets ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh oxford_pets ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh food101 ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh food101 ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh oxford_flowers ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh oxford_flowers ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh sun397 ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh sun397 ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh stanford_cars ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh stanford_cars ${SEED} cuda:1
  bash scripts/maple/base2new_train_maple2.sh caltech101 ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple2.sh caltech101 ${SEED} cuda:1
done