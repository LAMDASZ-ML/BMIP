for SEED in 1 2 3
do
#    bash scripts/bitp/xd_test_bitp.sh imagenetv2 ${SEED}
    bash scripts/bitp/xd_test_bitp.sh imagenet_sketch ${SEED}
#    bash scripts/bitp/xd_test_bitp.sh imagenet_a ${SEED}
#    bash scripts/bitp/xd_test_bitp.sh imagenet_r ${SEED}
done