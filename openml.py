from openml import evaluations
from openml import study
benchmark_suite = study.get_study('OpenML100','tasks') # obtain the benchmark suite

# Download and group the evaluation data
scores = []
for task_id in benchmark_suite.tasks: # iterate over all tasks. Can be removed if list_evaluations used paging!
    evaluations = evaluations.list_evaluations(task=[task_id], function='area_under_roc_curve', size=200)
    for id, e in evaluations.items():
        scores.append({"dataset":e.data_name, "score":e.value})
