"""
Simple test script to check if all modules can be imported.
Run this to verify the installation before training.
"""

import sys

def test_imports():
    """Test all module imports."""
    errors = []
    
    # Test basic imports
    print("Testing basic imports...")
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")
        print(f"  ✗ torch: {e}")
    
    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print(f"  ✗ numpy: {e}")
    
    try:
        import pandas as pd
        print(f"  ✓ pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"pandas: {e}")
        print(f"  ✗ pandas: {e}")
    
    # Test PyTorch Geometric
    print("\nTesting PyTorch Geometric...")
    try:
        import torch_geometric
        print(f"  ✓ torch_geometric {torch_geometric.__version__}")
    except ImportError as e:
        errors.append(f"torch_geometric: {e}")
        print(f"  ✗ torch_geometric: {e}")
    
    try:
        import torch_sparse
        print("  ✓ torch_sparse")
    except ImportError as e:
        errors.append(f"torch_sparse: {e}")
        print(f"  ✗ torch_sparse: {e}")
    
    # Test project modules
    print("\nTesting project modules...")
    try:
        from data.data_loader import HFRSDatasetFromPT, load_graph
        print("  ✓ data.data_loader")
    except ImportError as e:
        errors.append(f"data.data_loader: {e}")
        print(f"  ✗ data.data_loader: {e}")
    
    try:
        from models.mopi_hfrs import MOPI_HFRS, create_model
        print("  ✓ models.mopi_hfrs")
    except ImportError as e:
        errors.append(f"models.mopi_hfrs: {e}")
        print(f"  ✗ models.mopi_hfrs: {e}")
    
    try:
        from utils.losses import bpr_loss, health_loss, diversity_loss
        print("  ✓ utils.losses")
    except ImportError as e:
        errors.append(f"utils.losses: {e}")
        print(f"  ✗ utils.losses: {e}")
    
    try:
        from utils.pareto import MinNormSolver, pareto_loss
        print("  ✓ utils.pareto")
    except ImportError as e:
        errors.append(f"utils.pareto: {e}")
        print(f"  ✗ utils.pareto: {e}")
    
    try:
        from utils.metrics import get_metrics
        print("  ✓ utils.metrics")
    except ImportError as e:
        errors.append(f"utils.metrics: {e}")
        print(f"  ✗ utils.metrics: {e}")
    
    try:
        from config import get_default_config
        print("  ✓ config")
    except ImportError as e:
        errors.append(f"config: {e}")
        print(f"  ✗ config: {e}")
    
    # Summary
    print("\n" + "="*50)
    if errors:
        print(f"FAILED: {len(errors)} import errors")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("SUCCESS: All imports passed!")
        return True


if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)

