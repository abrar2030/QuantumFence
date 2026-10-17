import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_INTEG_DIR = os.path.dirname(_THIS_DIR)
_CODE_DIR = os.path.dirname(_INTEG_DIR)
_BACKEND_DIR = os.path.join(_CODE_DIR, "backend")

for _p in (_INTEG_DIR, _CODE_DIR, _BACKEND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("ANTHROPIC_API_KEY", "")
os.environ.setdefault("GOOGLE_MAPS_API_KEY", "")
os.environ.setdefault("SNAPSHOTS_DIR", "/tmp/qf_integ_test_snaps")
