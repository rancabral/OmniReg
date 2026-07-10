"""
OmniReg Cross-Platform Launcher
Works on: Windows PC, Linux, Raspberry Pi
"""
import sys
import platform
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("\n" + "="*75)
    print("  [OmniReg] Mathematical Equation Recognition System")
    print("="*75)
    
    # Detect platform
    system = platform.system()
    python_ver = platform.python_version()
    project_root = Path(__file__).parent
    
    print(f"\n  [OK] Platform: {system}")
    print(f"  [OK] Python Version: {python_ver}")
    print(f"  [OK] Project Location: {project_root.name}/")
    
    # Check core directories
    print(f"\n  [CHECKING] Core Directories:")
    
    dataset_path = project_root / "dataset"
    models_path = project_root / "models"
    pipeline_path = project_root / "pipeline_folder"
    nlp_path = project_root / "nlp_folder"
    gui_path = project_root / "gui_folder"
    
    checks = [
        ("Dataset", dataset_path),
        ("Models", models_path),
        ("Pipeline", pipeline_path),
        ("NLP", nlp_path),
        ("GUI", gui_path)
    ]
    
    for name, path in checks:
        if path.exists():
            print(f"     [OK] {name:12} -> Ready")
        else:
            print(f"     [!] {name:12} -> Missing")
    
    # Check dataset images
    if dataset_path.exists():
        images = list(dataset_path.glob("*.png"))
        print(f"\n  [INFO] Dataset: {len(images)} equation images loaded")
    else:
        print(f"\n  [!] Dataset folder not found at {dataset_path}")
    
    # Launch GUI
    print(f"\n" + "="*75)
    print("  [LAUNCH] Starting GUI Interface...")
    print("="*75 + "\n")
    
    try:
        from gui_folder.main_gui import OmniRegGUI
        import tkinter as tk
        
        root = tk.Tk()
        app = OmniRegGUI(root)
        root.mainloop()
    
    except ImportError as e:
        print(f"\n  [ERROR] Import Error: {e}")
        print(f"\n     Make sure all required packages are installed:")
        print(f"     • tkinter (built-in)")
        print(f"     • opencv-python")
        print(f"     • pillow")
        print(f"     • numpy")
        print(f"     • pix2tex")
        print(f"     • torch")
        print(f"     • torchvision")
        print(f"     • munch")
        print(f"\n     Run: pip install -r requirements.txt\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] Error launching GUI: {e}")
        print(f"\n     Try restarting the application.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
