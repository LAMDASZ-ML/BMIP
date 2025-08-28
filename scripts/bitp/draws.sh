DATA=$1
TRAINER=$2
CFG=$3
SUB=$4
for SEED in 1 2 3
do
  bash scripts/bitp/base2new_draw_bitp.sh ${DATA} ${SEED} ${TRAINER} ${CFG} ${SUB}
done
echo "Done with ${DATA} ${TRAINER} ${CFG} ${SUB}"