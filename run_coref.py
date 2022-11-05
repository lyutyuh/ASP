"""
    Training ASP for Coreference Resolution
    Tianyu Liu
"""
import sys
import logging
import random
import numpy as np

import torch

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import time
from os.path import join
from datetime import datetime

import util
from util.runner import Runner

from metrics import CorefEvaluator, MentionEvaluator


class CorefRunner(Runner):

    def evaluate(
        self, model, tensor_examples, stored_info, step, predict=False
    ):
        evaluator, mention_evaluator = CorefEvaluator(), MentionEvaluator()

        eval_batch_size = 4
        if any(substr in self.config["plm_pretrained_name_or_path"].lower()\
           for substr in ["pp", "11b"]):
            eval_batch_size = 4
        elif any(substr in self.config["plm_pretrained_name_or_path"].lower()\
             for substr in ["base"]):
            eval_batch_size = 8

        util.runner.logger.info('Step %d: evaluating on %d samples with batch_size %d' % (
            step, len(tensor_examples), eval_batch_size))

        evalloader = DataLoader(
            tensor_examples, batch_size=eval_batch_size, shuffle=False, 
            num_workers=0,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        model.eval()
        for i, (doc_keys, tensor_example) in enumerate(evalloader):
            example_gpu = {
                k: v.to(self.device) if v is not None else None for k, v in tensor_example.items()
            }

            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=self.use_amp, dtype=torch.bfloat16
            ):
                output = model(**example_gpu)

            for batch_id, doc_key in enumerate(doc_keys):

                gold_res = model.extract_gold_clusters_from_gold_annotation(
                    stored_info['example'][doc_key]
                )
                decoded_results = model.decoding(
                    {k:v[batch_id] for k,v in output.items()}, 
                    stored_info['example'][doc_key]
                )

                decoded_results.update(
                    gold_res
                )  # update gold clustering
                evaluator.update(
                    **decoded_results
                )
                mention_evaluator.update(
                    decoded_results["predicted_mentions"], 
                    gold_res["gold_mentions"],
                    len(stored_info['example'][doc_key]['sentence'])
                )

                if predict:  # logging results
                    util.runner.logger.info(stored_info['example'][doc_key]['sentence'])
                    util.runner.logger.info(decoded_results["predicted"])
                    util.runner.logger.info(decoded_results["gold"])

        p, r, f, blanc_prf = evaluator.get_prf()
        mention_recall = mention_evaluator.get_mention_recall()
        all_metrics = evaluator.get_all()

        metrics = {
            'Eval_Avg_Precision': p * 100,
            'Eval_Avg_Recall': r * 100,
            'Eval_Avg_F1': f * 100,
            'Eval_Mention_Recall': mention_recall * 100
        }
        for k, v in metrics.items():
            util.runner.logger.info('%s: %.4f' % (k, v))
        for k, v in all_metrics.items():
            util.runner.logger.info('%s: %.4f' % (k, v))

        return f * 100, metrics

# E.g.
# CUDA_VISIBLE_DEVICES=0 python run_coref.py t5_base 0

if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    saved_suffix = sys.argv[3] if len(sys.argv) >= 4 else None
    runner = CorefRunner(
        config_file="configs/coref.conf",
        config_name=config_name,
        gpu_id=gpu_id
    )

    if saved_suffix is not None:
        model, start_epoch = runner.initialize_model(saved_suffix, continue_training=True)
        runner.train(model, continued=True, start_epoch=start_epoch)
    else:
        model, _ = runner.initialize_model()
        runner.train(model, continued=False)
