import json
import argparse
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LongTail Dataset')
    parser.add_argument('--fn', metavar='PATH', type=str, help='path to the dataset json file', default="dataset.json")
    parser.add_argument('--lt_cls', type=int, help='index of the lt class')
    parser.add_argument('--lt_count', type=int, help='number of samples in the lt class')
    args = parser.parse_args()

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
    lt_pair = {"lables": pair_list}
    with open(os.path.join(pwd, f"pair-lt-{args.lt_cls}-{args.lt_count}.json"), "w") as f:
        json.dump(lt_pair, f)


    # with open(os.path.join(pwd, f"dataset-lt-{args.lt_cls}-{}-rem.json"), "w") as f:
    #     json.dump(lt_sk, f)

    # with open(os.path.join(pwd, "dataset_init.json"), "w") as f:
    #     json.dump(main_, f)

