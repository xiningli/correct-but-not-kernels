"""K1/K2: unbiased pass@k and finite-sample mixed-outcome GRPO group probability."""
import math
import statistics

def comb_ratio(successes, n, k):
    return math.comb(successes,k)/math.comb(n,k) if successes >= k else 0.0

def summarize(rows, group=8):
    by_task = {}
    for row in rows:
        by_task.setdefault(row['task'], []).append(row)
    tasks = {}
    for task, values in by_task.items():
        n = len(values)
        if n < group:
            raise ValueError('Insufficient samples for group signal')
        c = sum(v['result']['correct'] for v in values)
        speeds = [v['result']['speedup'] for v in values if v['result']['correct']]
        tasks[task] = {'n':n,'correct':c,'pass_at_1':c/n,
                       'pass_at_8':1-comb_ratio(n-c,n,group),
                       'p_signal':1-comb_ratio(c,n,group)-comb_ratio(n-c,n,group),
                       'median_tokens':statistics.median(v['tokens'] for v in values),
                       'truncation_fraction':sum(v['finish_reason']=='length' for v in values)/n,
                       'correct_speedups':speeds,
                       'harness_errors':sum(v['result']['component']=='harness_error' for v in values)}
    return {'tasks':tasks, 'mean_p_signal':statistics.fmean(v['p_signal'] for v in tasks.values()),
            'signal_tasks':sum(v['pass_at_8']>0 for v in tasks.values()),
            'harness_errors':sum(v['harness_errors'] for v in tasks.values())}
