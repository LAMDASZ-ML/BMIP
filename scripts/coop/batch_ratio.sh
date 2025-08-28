for SEED in 1 2 3
do
  for ((i=100; i>=0; i-=10))
  do
    bash scripts/coop/base2new_ratio.sh eurosat ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh fgvc_aircraft ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh caltech101 ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh food101 ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh dtd ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh imagenet ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh oxford_pets ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh oxford_flowers ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh stanford_cars ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh sun397 ${SEED} cuda:1 ${i}
    bash scripts/coop/base2new_ratio.sh ucf101 ${SEED} cuda:1 ${i}
  done
done