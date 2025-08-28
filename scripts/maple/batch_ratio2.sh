for SEED in 1 2 3
do
  for ((i=100; i>=0; i-=10))
  do
    bash scripts/maple/base2new_ratio2.sh eurosat ${SEED} cuda:1 ${i}
    bash scripts/maple/base2new_ratio2.sh fgvc_aircraft ${SEED} cuda:1 ${i}
    bash scripts/maple/base2new_ratio2.sh imagenet ${SEED} cuda:0 ${i}
    bash scripts/maple/base2new_ratio2.sh dtd ${SEED} cuda:0 ${i}
    bash scripts/maple/base2new_ratio2.sh ucf101 ${SEED} cuda:0 ${i}
    bash scripts/maple/base2new_ratio2.sh oxford_pets ${SEED} cuda:0 ${i}
    bash scripts/maple/base2new_ratio2.sh food101 ${SEED} cuda:0 ${i}
    bash scripts/maple/base2new_ratio2.sh oxford_flowers ${SEED} cuda:0 ${i}
    bash scripts/maple/base2new_ratio2.sh sun397 ${SEED} cuda:0 ${i}
    bash scripts/maple/base2new_ratio2.sh stanford_cars ${SEED} cuda:0 ${i}
    bash scripts/maple/base2new_ratio2.sh caltech101 ${SEED} cuda:0 ${i}
  done
done