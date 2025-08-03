from typing import List, Dict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def calc_metrics(preds: List[str], refs: List[str]) -> Dict:
    """
    计算 BLEU-1/2/3/4 与 ROUGE-1/2/L 的平均分。
    """
    # BLEU 准备
    smoothie = SmoothingFunction().method1
    bleu1, bleu2, bleu3, bleu4 = [], [], [], []

    # ROUGE 准备（非递归实现）
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                      use_stemmer=True)
    rouge1, rouge2, rouge_l = [], [], []

    for p, r in zip(preds, refs):
        # -------- BLEU --------
        bleu1.append(sentence_bleu([r.split()], p.split(),
                                   weights=(1, 0, 0, 0),
                                   smoothing_function=smoothie))
        bleu2.append(sentence_bleu([r.split()], p.split(),
                                   weights=(0.5, 0.5, 0, 0),
                                   smoothing_function=smoothie))
        bleu3.append(sentence_bleu([r.split()], p.split(),
                                   weights=(1/3, 1/3, 1/3, 0),
                                   smoothing_function=smoothie))
        bleu4.append(sentence_bleu([r.split()], p.split(),
                                   weights=(0.25, 0.25, 0.25, 0.25),
                                   smoothing_function=smoothie))

        # -------- ROUGE --------
        scores = scorer.score(r, p)         # 先参考文本，后预测文本
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rouge_l.append(scores["rougeL"].fmeasure)

    # -------- 汇总结果 --------
    return {
        "BLEU-1":  {"average": sum(bleu1)  / len(bleu1)},
        "BLEU-2":  {"average": sum(bleu2)  / len(bleu2)},
        "BLEU-3":  {"average": sum(bleu3)  / len(bleu3)},
        "BLEU-4":  {"average": sum(bleu4)  / len(bleu4)},
        "ROUGE-1": {"average": sum(rouge1) / len(rouge1)},
        "ROUGE-2": {"average": sum(rouge2) / len(rouge2)},
        "ROUGE-L": {"average": sum(rouge_l)/ len(rouge_l)},
    }
