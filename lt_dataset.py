import json
import argparse
import numpy as np
import os


def base_target_lt(args):
    pwd = os.path.dirname(args.fn)

    with open(args.fn) as f:
        main_ = json.load(f)

    if main_["labels"]:
        main = np.array(main_["labels"], dtype=object)
    else:
        exit(0)

    all_inds = np.arange(main.shape[0])
    lt_class_inds = np.where(main[:, 1] == args.lt_cls)[0]
    s_inds = lt_class_inds[:args.lt_count]
    ns_inds = lt_class_inds[args.lt_count:]
    # s_lt_inds = np.random.choice(lt_inds, size=args.lt_count, replace=False)
    # ns_lt_inds = np.array(list(set(lt_inds) - set(s_lt_inds)))
    new_inds = list(set(all_inds) - set(ns_inds))

    lt = {"labels": main[new_inds].tolist()}
    lt_s = {"labels": main[s_inds].tolist()}
    # lt_sk = {"labels": main[ns_inds].tolist()}

    with open(os.path.join(pwd, f"dataset-lt-{args.lt_cls}-{args.lt_count}.json"), "w") as f:
        json.dump(lt, f)

    with open(os.path.join(pwd, f"only-lt-{args.lt_cls}-{args.lt_count}.json"), "w") as f:
        json.dump(lt_s, f)

    pair_list = [[v[0], v[1] if v[1] == args.lt_cls else -1] for v in main[new_inds].tolist()]
    lt_pair = {"labelvim s": pair_list}
    with open(os.path.join(pwd, f"pair-lt-{args.lt_cls}-{args.lt_count}.json"), "w") as f:
        json.dump(lt_pair, f)


def dataset_lt(args):
    pwd = os.path.dirname(args.fn)
    with open(args.fn) as f:
        main_ = json.load(f)

    if main_["labels"]:
        main = np.array(main_["labels"], dtype=object)
    else:
        exit(0)

    img_num_per_cls, classes = get_img_num_per_cls(main, args)
    lt_ds = gen_imbalanced_data(main, img_num_per_cls, classes)
    lt = {"labels": lt_ds.tolist()}

    filename = f"lt_{args.imf}"
    if args.reverse:
        filename += "_reverse"
    with open(os.path.join(pwd, f"{filename}.json"), "w") as f:
        json.dump(lt, f)


def get_img_num_per_cls(ds, args):
    classes = dict()
    for d in ds:
        if d[1] in classes:
            classes[d[1]] += 1
        else:
            classes[d[1]] = 1
    cls_num = len(classes)
    # img_max = classes[max(classes, key=classes.get)]
    img_max = ds.shape[0] // cls_num

    img_num_per_cls = []
    for cls_idx in set(classes):
        if args.reverse:
            num = img_max * (1/args.imf ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        else:
            num = img_max * (1/args.imf ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))

    return img_num_per_cls, set(classes)


def gen_imbalanced_data(ds, img_num_per_cls, classes):
    new_data = []
    for i, count in zip(classes, img_num_per_cls):
        idx = np.where(ds[:, 1] == i)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:count]
        new_data.append(ds[selec_idx, ...])
    return np.vstack(new_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LongTail Dataset')
    parser.add_argument('--fn', metavar='PATH', type=str, help='path to the dataset json file', default="dataset.json")
    parser.add_argument('--lt_cls', type=int, default=None, help='index of the lt class')
    parser.add_argument('--lt_count', type=int, default=25, help='number of samples in the lt class')
    parser.add_argument('--imf', type=int, default=100, help='number of samples in the lt class')
    parser.add_argument('--reverse', action="store_true", help='number of samples in the lt class')
    args = parser.parse_args()

    if args.lt_cls:
        base_target_lt(args)
    else:
        dataset_lt(args)




    # with open(os.path.join(pwd, f"dataset-lt-{args.lt_cls}-{}-rem.json"), "w") as f:
    #     json.dump(lt_sk, f)

    # with open(os.path.join(pwd, "dataset_init.json"), "w") as f:
    #     json.dump(main_, f)


    # #------------------------------------------------------------------------------------
    # # From: https://github.com/kaidic/LDAM-DRW
    # import torch
    # import torchvision
    # import torchvision.transforms as transforms
    # import numpy as np
    #
    #
    # class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    #     cls_num = 10
    #
    #     def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
    #                  transform=None, target_transform=None,
    #                  download=False, reverse=False):
    #         super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
    #         np.random.seed(rand_number)
    #         img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
    #         self.gen_imbalanced_data(img_num_list)
    #         self.reverse = reverse
    #
    #     def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
    #         img_max = len(self.data) / cls_num
    #         img_num_per_cls = []
    #         if imb_type == 'exp':
    #             for cls_idx in range(cls_num):
    #                 if reverse:
    #                     num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
    #                     img_num_per_cls.append(int(num))
    #                 else:
    #                     num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
    #                     img_num_per_cls.append(int(num))
    #         elif imb_type == 'step':
    #             for cls_idx in range(cls_num // 2):
    #                 img_num_per_cls.append(int(img_max))
    #             for cls_idx in range(cls_num // 2):
    #                 img_num_per_cls.append(int(img_max * imb_factor))
    #         else:
    #             img_num_per_cls.extend([int(img_max)] * cls_num)
    #         return img_num_per_cls
    #
    #     def gen_imbalanced_data(self, img_num_per_cls):
    #         new_data = []
    #         new_targets = []
    #         targets_np = np.array(self.targets, dtype=np.int64)
    #         classes = np.unique(targets_np)
    #         # np.random.shuffle(classes)
    #         self.num_per_cls_dict = dict()
    #         for the_class, the_img_num in zip(classes, img_num_per_cls):
    #             self.num_per_cls_dict[the_class] = the_img_num
    #             idx = np.where(targets_np == the_class)[0]
    #             np.random.shuffle(idx)
    #             selec_idx = idx[:the_img_num]
    #             new_data.append(self.data[selec_idx, ...])
    #             new_targets.extend([the_class, ] * the_img_num)
    #         new_data = np.vstack(new_data)
    #         self.data = new_data
    #         self.targets = new_targets
    #
    #     def get_cls_num_list(self):
    #         cls_num_list = []
    #         for i in range(self.cls_num):
    #             cls_num_list.append(self.num_per_cls_dict[i])
    #         return cls_num_list
    #
    #
    # class IMBALANCECIFAR100(IMBALANCECIFAR10):
    #     """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    #     This is a subclass of the `CIFAR10` Dataset.
    #     """
    #     base_folder = 'cifar-100-python'
    #     url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    #     filename = "cifar-100-python.tar.gz"
    #     tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    #     train_list = [
    #         ['train', '16019d7e3df5f24257cddd939b257f8d'],
    #     ]
    #
    #     test_list = [
    #         ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    #     ]
    #     meta = {
    #         'filename': 'meta',
    #         'key': 'fine_label_names',
    #         'md5': '7973b15100ade9c7d40fb424638fde48',
    #     }
    #     cls_num = 100
    #
    #
    # if __name__ == '__main__':
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #     trainset = IMBALANCECIFAR100(root='./data', train=True,
    #                                  download=True, transform=transform)
    #     trainloader = iter(trainset)
    #     data, label = next(trainloader)
    #     import pdb;
    #
    #     pdb.set_trace()