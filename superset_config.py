import os

SECRET_KEY = os.environ.get(
    "SUPERSET_SECRET_KEY",
    "fallback_secret"
)

PREVENT_UNSAFE_DB_CONNECTIONS = False

FEATURE_FLAGS = {
    "ENABLE_UPLOAD_CSV": True
}