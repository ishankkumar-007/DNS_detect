"""
Quick Test for Flask App Setup
Tests that all components are ready
"""

import sys
from pathlib import Path

def test_imports():
    """Test required imports"""
    print("Testing imports...")
    
    try:
        import flask
        print(f"  ✓ Flask {flask.__version__}")
    except ImportError as e:
        print(f"  ✗ Flask: {e}")
        return False
    
    try:
        import pandas
        print(f"  ✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas: {e}")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    try:
        import lightgbm
        print(f"  ✓ LightGBM {lightgbm.__version__}")
    except ImportError as e:
        print(f"  ✗ LightGBM: {e}")
        return False
    
    return True

def test_directories():
    """Test required directories exist"""
    print("\nTesting directories...")
    
    base_dir = Path(__file__).parent
    
    # Check templates
    templates_dir = base_dir / 'templates'
    if templates_dir.exists() and (templates_dir / 'index.html').exists():
        print(f"  ✓ Templates directory: {templates_dir}")
    else:
        print(f"  ✗ Templates directory or index.html missing: {templates_dir}")
        return False
    
    # Check results directory
    results_dir = base_dir / 'results'
    if results_dir.exists():
        model_count = sum(1 for d in results_dir.iterdir() if d.is_dir())
        print(f"  ✓ Results directory: {results_dir} ({model_count} model directories)")
    else:
        print(f"  ⚠ Results directory not found: {results_dir}")
        print("    Note: App will work but no models will be available")
    
    # Create uploads directory
    uploads_dir = base_dir / 'uploads'
    uploads_dir.mkdir(exist_ok=True)
    print(f"  ✓ Uploads directory: {uploads_dir}")
    
    return True

def test_model_discovery():
    """Test model discovery"""
    print("\nTesting model discovery...")
    
    results_dir = Path(__file__).parent / 'results'
    if not results_dir.exists():
        print("  ⚠ No results directory - skipping model discovery")
        return True
    
    models_found = 0
    for result_dir in results_dir.iterdir():
        if not result_dir.is_dir():
            continue
        
        models_subdir = result_dir / 'models'
        if models_subdir.exists():
            model_files = list(models_subdir.glob('*_detector.*'))
            if model_files:
                print(f"  ✓ {result_dir.name}: {len(model_files)} model file(s)")
                models_found += 1
    
    print(f"\nTotal: {models_found} model directories with detector files")
    return True

def main():
    print("="*60)
    print("FLASK APP SETUP TEST")
    print("="*60 + "\n")
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_directories():
        all_passed = False
    
    if not test_model_discovery():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYou can now start the Flask app:")
        print("  python app.py")
        print("\nOr use the start script:")
        print("  python start_flask_app.py")
        print("\nThen open: http://localhost:5000")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before starting the app.")
    print("="*60)

if __name__ == "__main__":
    main()
