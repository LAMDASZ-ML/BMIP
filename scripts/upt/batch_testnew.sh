for SEED in 1 2 3
do
bash scripts/maple/base2new_test_maple.sh dtd ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh ucf101 ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh fgvc_aircraft ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh oxford_pets ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh food101 ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh eurosat ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh oxford_flowers ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh sun397 ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh stanford_cars ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh caltech101 ${SEED} cuda:0 
bash scripts/maple/base2new_test_maple.sh imagenet ${SEED} cuda:0 
done