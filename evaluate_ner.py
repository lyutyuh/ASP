from run_ner import NERRunner
import sys
import torch


def evaluate(config_name, gpu_id, saved_suffix):
    runner = NERRunner(
        config_file="configs/ner.conf",
        config_name=config_name,
        gpu_id=gpu_id
    )
    model, _ = runner.initialize_model(saved_suffix, continue_training=False)

    examples_train, examples_dev, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    runner.evaluate(model, examples_test, stored_info, 0, predict=True)  # Eval test


# E.g.
# CUDA_VISIBLE_DEVICES=0 python evaluate_ner.py t5_base Aug14_19-53-06_85000 0
if __name__ == '__main__':
    config_name, saved_suffix, gpu_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
    evaluate(config_name, gpu_id, saved_suffix)
