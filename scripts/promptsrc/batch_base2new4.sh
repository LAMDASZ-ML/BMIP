CFG=$1
for SEED in 1 2 3
do
  # bash scripts/promptsrc/base2new_train_promptsrc4.sh eurosat ${SEED} cuda:1 ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc4.sh eurosat ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc4.sh fgvc_aircraft ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc4.sh fgvc_aircraft ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc4.sh dtd ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc4.sh  dtd ${SEED} cuda:1 ${CFG}
  # bash scripts/promptsrc/base2new_train_promptsrc4.sh ucf101 ${SEED} cuda:1 ${CFG}
  # bash scripts/promptsrc/base2new_test_promptsrc4.sh  ucf101 ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc4.sh oxford_pets ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc4.sh  oxford_pets ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc4.sh food101 ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc4.sh  food101 ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc4.sh oxford_flowers ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc4.sh  oxford_flowers ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc4.sh sun397 ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc4.sh  sun397 ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc4.sh stanford_cars ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc4.sh  stanford_cars ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_train_promptsrc4.sh caltech101 ${SEED} cuda:1 ${CFG}
  bash scripts/promptsrc/base2new_test_promptsrc4.sh  caltech101 ${SEED} cuda:1 ${CFG}
done