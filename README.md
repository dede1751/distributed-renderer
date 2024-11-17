# BlenderProc Objaverse Multiview Renderer

Setup the environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Downloading Objaverse Assets

```bash
python utils/download.py --num_objects 100 --num_workers 8 --list_file obj_list.txt
```


## Downloading HDRIs

```bash
blenderproc download haven --categories skies pure%20skies high%20contrast --resolution 1k hdri/high_contrast
```