for SEED in 1 2 3
do
#  bash scripts/cocoop/base2new_train.sh fgvc_aircraft ${SEED}
#  bash scripts/cocoop/base2new_test.sh fgvc_aircraft ${SEED}
  bash scripts/cocoop/base2new_train.sh eurosat ${SEED} 
  bash scripts/cocoop/base2new_test.sh eurosat ${SEED} 
#  bash scripts/cocoop/base2new_train.sh imagenet ${SEED}
#  bash scripts/cocoop/base2new_test.sh imagenet ${SEED}
#  bash scripts/cocoop/base2new_train.sh dtd ${SEED}
#  bash scripts/cocoop/base2new_test.sh  dtd ${SEED}
#  bash scripts/cocoop/base2new_train.sh ucf101 ${SEED}
#  bash scripts/cocoop/base2new_test.sh  ucf101 ${SEED}
#  bash scripts/cocoop/base2new_train.sh oxford_pets ${SEED}
#  bash scripts/cocoop/base2new_test.sh  oxford_pets ${SEED}
#  bash scripts/cocoop/base2new_train.sh food101 ${SEED}
#  bash scripts/cocoop/base2new_test.sh  food101 ${SEED}
#  bash scripts/cocoop/base2new_train.sh oxford_flowers ${SEED}
#  bash scripts/cocoop/base2new_test.sh  oxford_flowers ${SEED}
#  bash scripts/cocoop/base2new_train.sh sun397 ${SEED}
#  bash scripts/cocoop/base2new_test.sh  sun397 ${SEED}
#  bash scripts/cocoop/base2new_train.sh stanford_cars ${SEED}
#  bash scripts/cocoop/base2new_test.sh  stanford_cars ${SEED}
#  bash scripts/cocoop/base2new_train.sh caltech101 ${SEED}
#  bash scripts/cocoop/base2new_test.sh  caltech101 ${SEED}
done