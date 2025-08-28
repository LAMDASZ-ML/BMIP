for SEED in 1 2 3
do
  bash scripts/promptsrc/base2new_train_promptsrc1.sh eurosat ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh eurosat ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh fgvc_aircraft ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh fgvc_aircraft ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh dtd ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh  dtd ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh ucf101 ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh  ucf101 ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh oxford_pets ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh  oxford_pets ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh food101 ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh  food101 ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh oxford_flowers ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh  oxford_flowers ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh sun397 ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh  sun397 ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh stanford_cars ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh  stanford_cars ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh caltech101 ${SEED} cuda:0
  bash scripts/promptsrc/base2new_test_promptsrc1.sh  caltech101 ${SEED} cuda:0
  bash scripts/promptsrc/base2new_train_promptsrc1.sh imagenet ${SEED} cuda:0 ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc1.sh imagenet ${SEED} cuda:0 ${CFG}
done
