# BlenderProc Objaverse Multiview Renderer

Setup the environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Downloading Objaverse Assets

When running on the Euler cluster, modify the ```DATA_PATH``` variable in ```objaverse/__init__.py``` (inside the virtual environment's ```lib``` folder) to avoid caching meshes in the HOME directory, which has a limited capacity. Then run the following to download and process the dataset:

```bash
python utils/download.py --data_path <path/to/dataset> --num_objects 100 --num_workers 32 --list_file obj_list.txt
```

## Downloading HDRIs

```bash
blenderproc download haven --categories skies pure%20skies high%20contrast midday --resolution 1k hdri/midday
```