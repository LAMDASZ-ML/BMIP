import os
import argparse
import sys


#########################################
###            python cmd
#########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Elevater', add_help=False)
    parser.add_argument("--dataset_idx", type=int, default=13, help='')
    parser.add_argument("--dataset", type=str, default=None, help='')
    parser.add_argument("--out_path", type=str, default='./output', help='')
    parser.add_argument("--data_path", type=str, default=None, help='')
    parser.add_argument("--ckpt_path", type=str, default=None, help='')
    parser.add_argument("--init_method", type=str, default=None, help='')

    args = parser.parse_args()

    file_path = os.path.dirname(os.path.abspath(__file__))
    sh_path = os.path.join(file_path, 'run_proda.sh')

    args.dataset, dataset_path = [
        ('caltech101', 'caltech_101_20211007'),
        ('cifar10', 'cifar_10_20211007'),
        ('cifar100', 'cifar100_20200721'),
        ('country211', 'country211_20210924'),
        ('dtd','dtd_20211007'),
        ('eurosat-clip', 'eurosat_clip_20210930'),
        ('fer2013','fer_2013_20211008'),
        ('fgvc-aircraft-2013b', 'fgvc_aircraft_2013b_variants102_20211007'),
        ('flower102', 'oxford_flower_102_20211007'),
        ('food101', 'food_101_20211007'),
        ('gtsrb', 'gtsrb_20210923'), 
        ('hateful-memes', 'hateful_memes_20211014'),
        ('kitti-distance', 'kitti_distance_20210923'),
        ('mnist', 'mnist_20211008'),
        ('oxford-iiit-pets', 'oxford_iiit_pets_20211007'),
        ('patchcamelyon', 'patch_camelyon_20211007'),
        ('rendered-sst2', 'rendered_sst2_20210924'),
        ('resisc45-clip', 'resisc45_clip_20210924'),
        ('stanfordcar', 'stanford_cars_20211007'),
        ('voc2007classification', 'voc2007_20211007')][args.dataset_idx]


    for seed in [1,2,3]:
        sh_cmd = 'sh {} {} {} {} {} {} {}'.format(
            sh_path, args.dataset, seed, args.out_path, args.data_path, args.ckpt_path, file_path)
        print('exect shell cmd: {}'.format(sh_cmd))
        os.system(sh_cmd)
    print('Done!')


