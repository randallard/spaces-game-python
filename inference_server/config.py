"""
Configuration for the inference server.

All settings are configurable via environment variables.
"""

import os

INFERENCE_MODELS_DIR = os.environ.get('INFERENCE_MODELS_DIR', 'inference_server/models/')
INFERENCE_BOARDS_DIR = os.environ.get('INFERENCE_BOARDS_DIR', 'boards/')
INFERENCE_HOST = os.environ.get('INFERENCE_HOST', '0.0.0.0')
# Railway sets PORT; fall back to INFERENCE_PORT, then default 8100
INFERENCE_PORT = int(os.environ.get('PORT', os.environ.get('INFERENCE_PORT', '8100')))
INFERENCE_CORS_ORIGINS = os.environ.get(
    'INFERENCE_CORS_ORIGINS', 'http://localhost:5173'
).split(',')
