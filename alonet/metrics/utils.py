from typing import Dict, List


def _print_map(average_pq: Dict, pq_per_class: Dict, suffix: str = "", **kwargs):
    _print_head(suffix, **kwargs)
    _print_body(average_pq, pq_per_class, **kwargs)


def _print_head(suffix: str = "", head_elm: List = [], clm_size: int = 9, **kwargs):
    make_row = lambda vals: (f" %{clm_size - 2}s |" * len(vals)) % tuple(vals)
    make_sep = lambda n: (("-" * clm_size + "+") * (n + 1))

    print()
    print(make_sep(len(head_elm) + 1))
    print(" " * (2 * clm_size + 1) + "|" + make_row([v + suffix for v in head_elm]))
    print(make_sep(len(head_elm) + 1))


def _print_body(average_pq: Dict, pq_per_class: Dict, clm_size: int = 9, **kwargs):
    make_row = lambda vals: (f" %{clm_size - 2}s |" * len(vals)) % tuple(vals)
    make_sep = lambda n: (("-" * clm_size + "+") * (n + 1))

    if pq_per_class is not None:
        for cat, metrics in pq_per_class.items():
            print(
                make_row(
                    [cat[: clm_size * 2 - 1] if len(cat) > clm_size * 2 else cat + " " * (clm_size * 2 - 1 - len(cat))]
                    + ["%.3f" % metrics[k] for k in metrics.keys()]
                )
            )
        print(make_sep(len(metrics) + 1))

    if average_pq is not None:
        n = "%d" % average_pq.pop("n")
        print(
            make_row(
                ["total = %s" % n + " " * (clm_size * 2 - 9 - len(n))]
                + ["%.3f" % average_pq[k] for k in average_pq.keys()]
            )
        )
        print(make_sep(len(average_pq) + 1))
