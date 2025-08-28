for SEED in 1 2 3
do
    bash scripts/maple/reproduce_maple.sh caltech101 ${SEED} /mnt/hdd/lvsl/base2new/caltech101
    bash scripts/maple/reproduce_maple.sh dtd ${SEED} /mnt/hdd/lvsl/base2new/dtd
    bash scripts/maple/reproduce_maple.sh eurosat ${SEED} /mnt/hdd/lvsl/base2new/eurosat
    bash scripts/maple/reproduce_maple.sh fgvc_aircraft ${SEED} /mnt/hdd/lvsl/base2new/fgvc_aircraft
    bash scripts/maple/reproduce_maple.sh food101 ${SEED} /mnt/hdd/lvsl/base2new/food101
    bash scripts/maple/reproduce_maple.sh imagenet ${SEED} /mnt/hdd/lvsl/base2new/imagenet
    bash scripts/maple/reproduce_maple.sh oxford_flowers ${SEED} /mnt/hdd/lvsl/base2new/oxford_flowers
    bash scripts/maple/reproduce_maple.sh oxford_pets ${SEED} /mnt/hdd/lvsl/base2new/oxford_pets
    bash scripts/maple/reproduce_maple.sh stanford_cars ${SEED} /mnt/hdd/lvsl/base2new/stanford_cars
    bash scripts/maple/reproduce_maple.sh sun397 ${SEED} /mnt/hdd/lvsl/base2new/sun397
    bash scripts/maple/reproduce_maple.sh ucf101 ${SEED} /mnt/hdd/lvsl/base2new/ucf101
done