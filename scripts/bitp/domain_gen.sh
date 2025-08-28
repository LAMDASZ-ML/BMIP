for SEED in 1 2 3
do
    bash scripts/bitp/xd_test_bitp.sh imagenetv2 ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh imagenet_sketch ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh imagenet_a ${SEED} cuda:4
    bash scripts/bitp/xd_test_bitp.sh imagenet_r ${SEED} cuda:4
done