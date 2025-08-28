for SEED in 1 2 3
do
    bash scripts/cocoop/base2new_train.sh aid ${SEED}
    bash scripts/cocoop/base2new_test.sh aid ${SEED}
    bash scripts/cocoop/base2new_train.sh mlrsnet ${SEED} 
    bash scripts/cocoop/base2new_test.sh mlrsnet ${SEED} 
    bash scripts/cocoop/base2new_train.sh optimal ${SEED}
    bash scripts/cocoop/base2new_test.sh optimal ${SEED}
    bash scripts/cocoop/base2new_train.sh patternnet ${SEED}
    bash scripts/cocoop/base2new_test.sh patternnet ${SEED}
    bash scripts/cocoop/base2new_train.sh resisc45 ${SEED}
    bash scripts/cocoop/base2new_test.sh  resisc45 ${SEED}
    bash scripts/cocoop/base2new_train.sh rsicb128 ${SEED}
    bash scripts/cocoop/base2new_test.sh  rsicb128 ${SEED}
    bash scripts/cocoop/base2new_train.sh rsicb256 ${SEED}
    bash scripts/cocoop/base2new_test.sh  rsicb256 ${SEED}
    bash scripts/cocoop/base2new_train.sh rsicd ${SEED}
    bash scripts/cocoop/base2new_test.sh  rsicd ${SEED}
    bash scripts/cocoop/base2new_train.sh rsitmd ${SEED}
    bash scripts/cocoop/base2new_test.sh  rsitmd ${SEED}
    bash scripts/cocoop/base2new_train.sh ucm ${SEED}
    bash scripts/cocoop/base2new_test.sh  ucm ${SEED}
    bash scripts/cocoop/base2new_train.sh whurs19 ${SEED}
    bash scripts/cocoop/base2new_test.sh whurs19 ${SEED}
done