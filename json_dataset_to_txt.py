import json, os, argparse
import pathlib as pl


def convert_json_dataset_to_txt(root: pl.Path):
    old_entity2id_path = root / "entity2id.json"
    with open(old_entity2id_path) as f:
        entity2id = json.load(f)
    old_relation2id_path = root / "relation2id.json"
    with open(old_relation2id_path) as f:
        rel2id = json.load(f)
    ts2id_path = root / "ts2id.json"
    with open(ts2id_path) as f:
        ts2id = json.load(f)

    train_path = root / "train.txt"
    valid_path = root / "valid.txt"
    test_path = root / "test.txt"
    for path in [train_path, valid_path, test_path]:
        new_lines = []
        with open(path) as f:
            for line in f:
                subj, rel, obj, ts = line.rstrip("\n").split("\t")
                subj_id = entity2id[subj]
                rel_id = rel2id[rel]
                obj_id = entity2id[obj]
                ts_id = ts2id[ts]
                new_lines.append(f"{subj_id}\t{rel_id}\t{obj_id}\t{ts_id}")
        with open(path, "w") as f:
            print(f"writing {path}...", end="")
            f.write("\n".join(new_lines))
            print("done!")

    stat_path = root / "stat.txt"
    with open(stat_path, "w") as f:
        print(f"writing {stat_path}...", end="")
        f.write(f"{len(entity2id)}\t{len(rel2id)}\t{len(ts2id)}")
        print("done!")

    new_entity2id_path = root / "entity2id.txt"
    with open(new_entity2id_path, "w") as f:
        print(f"writing {new_entity2id_path}...", end="")
        for entity, eid in entity2id.items():
            f.write(f"{entity}\t{eid}\n")
        print("done!")
    new_relation2id_path = root / "relation2id.txt"
    with open(new_relation2id_path, "w") as f:
        print(f"writing {new_relation2id_path}...", end="")
        for rel, rid in rel2id.items():
            f.write(f"{rel}\t{rid}\n")
        print("done!")

    print(f"deleting {old_relation2id_path}...", end="")
    os.remove(old_relation2id_path)
    print("done!")
    print(f"deleting {old_entity2id_path}...", end="")
    os.remove(old_entity2id_path)
    print("done!")
    print(f"deleting {ts2id_path}...", end="")
    os.remove(ts2id_path)
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=pl.Path)
    args = parser.parse_args()

    convert_json_dataset_to_txt(args.dataset)
