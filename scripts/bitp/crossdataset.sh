for SEED in 1 2 3
do
    bash scripts/bitp/xd_test_bitp.sh food101 ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh caltech101 ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh oxford_flowers ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh oxford_pets ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh stanford_cars ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh fgvc_aircraft ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh sun397 ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh eurosat ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh ucf101 ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh dtd ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh imagenet ${SEED} cuda:4
done