for SEED in 1 2 3
do
    bash scripts/maple/xd_test_maple.sh food101 ${SEED}
    bash scripts/maple/xd_test_maple.sh caltech101 ${SEED}
    bash scripts/maple/xd_test_maple.sh oxford_flowers ${SEED}
    bash scripts/maple/xd_test_maple.sh oxford_pets ${SEED}
    bash scripts/maple/xd_test_maple.sh stanford_cars ${SEED}
    bash scripts/maple/xd_test_maple.sh fgvc_aircraft ${SEED}
    bash scripts/maple/xd_test_maple.sh sun397 ${SEED}
    bash scripts/maple/xd_test_maple.sh eurosat ${SEED}
    bash scripts/maple/xd_test_maple.sh ucf101 ${SEED}
    bash scripts/maple/xd_test_maple.sh dtd ${SEED}
    bash scripts/maple/xd_test_maple.sh imagenet ${SEED}
done