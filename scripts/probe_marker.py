"""Quick probe: run marker on the test PDF and print the output structure."""

import os
import sys

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pdf = "data/test_multimodal.pdf"

print("[marker] creating model dict (downloads models on first run) ...")
model_dict = create_model_dict()

print("[marker] converting ...")
converter = PdfConverter(artifact_dict=model_dict)
rendered = converter(pdf)

print("[marker] output type:", type(rendered).__name__)
markdown, ext, images = text_from_rendered(rendered)
print("[marker] ext:", ext)
print("[marker] images dict keys:", list(images.keys()) if images else "none")
print("=" * 60)
print(markdown)
print("=" * 60)

from marker.models import shutdown_models
shutdown_models(model_dict)