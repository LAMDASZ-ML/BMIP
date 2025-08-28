for SEED in 1 2 3
do
  bash scripts/bitp/base2new_test_bitp_all.sh eurosat ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh dtd ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh caltech101 ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh fgvc_aircraft ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh imagenet ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh ucf101 ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh oxford_flowers ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh food101 ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh stanford_cars ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh sun397 ${SEED} cuda:2
  bash scripts/bitp/base2new_test_bitp_all.sh oxford_pets ${SEED} cuda:2
done