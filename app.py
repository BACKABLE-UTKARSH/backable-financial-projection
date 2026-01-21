# Import the main application from the full module name using importlib
import importlib.util
import sys

# Load the module with spaces in its name
spec = importlib.util.spec_from_file_location(
    "financial_projection_module",
    "BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules["financial_projection_module"] = module
spec.loader.exec_module(module)

# Get the app from the module
app = module.app

# Export for gunicorn
__all__ = ['app']
