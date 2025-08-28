for SEED in 1 2 3
do
#  bash scripts/maple/base2new_train_maple.sh fgvc_aircraft ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple.sh fgvc_aircraft ${SEED} cuda:0
#  bash scripts/maple/base2new_train_maple.sh eurosat ${SEED} cuda:0
  bash scripts/maple/base2new_test_maple.sh eurosat ${SEED} cuda:0
#  bash scripts/maple/base2new_train_maple.sh oxford_flowers ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple.sh oxford_flowers ${SEED} cuda:1
#  bash scripts/maple/base2new_train_maple.sh imagenet ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple.sh imagenet ${SEED} cuda:1
#  bash scripts/maple/base2new_train_maple.sh dtd ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple.sh dtd ${SEED} cuda:1
#  bash scripts/maple/base2new_train_maple.sh ucf101 ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple.sh ucf101 ${SEED} cuda:1
#  bash scripts/maple/base2new_train_maple.sh oxford_pets ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple.sh oxford_pets ${SEED} cuda:1
#  bash scripts/maple/base2new_train_maple.sh food101 ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple.sh food101 ${SEED} cuda:1
#  bash scripts/maple/base2new_train_maple.sh sun397 ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple.sh sun397 ${SEED} cuda:1
#  bash scripts/maple/base2new_train_maple.sh stanford_cars ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple.sh stanford_cars ${SEED} cuda:1
#  bash scripts/maple/base2new_train_maple.sh caltech101 ${SEED} cuda:1
  bash scripts/maple/base2new_test_maple.sh caltech101 ${SEED} cuda:1
done