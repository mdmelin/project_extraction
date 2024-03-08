# project_extraction
Extraction code for IBL satellite projects e.g. U19 / personal projects

## Installation
Clone the repository, and install in place
```
pip install -e .
```

## Reference

- `projects/extractor_types.json` - (DEPRECATED) For adding custom extractors for legacy pipeline tasks
- `projects/task_extractor_map.json` - Map custom task protocols to Bpod trials extractor class
- `projects/task_type_procedures.json` - Associate Alyx procedures to a custom task protocol
- `projects/_template.py` - Example for creating a custom Bpod extractor, QC or DAQ sync task
- `projects/_extraction_tasks.py` - Where to import pipeline tasks so they are readable by ibllib
