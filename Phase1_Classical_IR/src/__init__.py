"""
src/__init__.py
---------------
Makes src/ a Python package so notebooks can do:

    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    from src.data_loader import load_dataset
    from src.evaluation  import evaluate_rankings
    from src.visualization import plot_dataset_overview
"""
