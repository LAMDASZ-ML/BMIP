data=$1
data2=$2
for ((INIT_SEED=21; INIT_SEED<=36; INIT_SEED+=1))
do
  for SEED in 1 2 3
  do
    bash scripts/promptsrc/base2new_train_promptsrc.sh ${data} ${SEED} cuda:1 ${INIT_SEED}
    bash scripts/promptsrc/base2new_test_promptsrc.sh ${data} ${SEED} cuda:1 ${INIT_SEED}
    if [ -n "$data2" ]; then
      bash scripts/promptsrc/base2new_train_promptsrc.sh ${data2} ${SEED} cuda:1 ${INIT_SEED}
      bash scripts/promptsrc/base2new_test_promptsrc.sh ${data2} ${SEED} cuda:1 ${INIT_SEED}
    fi
  done
done