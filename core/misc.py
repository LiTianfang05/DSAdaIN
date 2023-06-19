import json
import os
import shutil


def arg2bool(arg):
    arg = arg.lower()
    if arg in ('1', 't', 'true'):
        return True
    elif arg in ('0', 'f', 'false'):
        return False
    else:
        raise ValueError


def arg2int(arg):
    arg = arg.lower()
    if arg.endswith("k"):
        arg = float(arg.replace("k", "")) * 1000
    elif arg.endswith("m"):
        arg = float(arg.replace("m", "")) * 1000000
    return int(arg)


def copy_py_files(src_dir, dst_dir, ignores=("logs", "runs")):
    for path, _, filenames in os.walk(src_dir):
        if not filenames:
            continue
        ignore = False
        for ig in ignores:
            if ig in path:
                ignore = True
                break
        if ignore:
            continue
        pyfiles = list(filter(lambda f: f.endswith(".py"), filenames))
        for file in pyfiles:
            src = os.path.join(src_dir, path, file)
            dst = os.path.join(dst_dir, path.replace(src_dir, "."))
            os.makedirs(dst, exist_ok=True)
            shutil.copyfile(src, os.path.join(dst, file))


def report_metric(result_dict, run_dir=None):
    jsonl_line = json.dumps(result_dict)
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')
