# BlenderProc Multiview Distributed Renderer

Setup the environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
blenderproc pip install psutil
```

## Downloading Objaverse Assets

Note that the cache directory for Objaverse assets has been moved in ```datasets/objaverse_json.py``` to avoid cluster storage limits, feel free to comment that out.
To prepare a list of Objaverse UIDs for rendering, simply run:

```bash
python datasets/objaverse_json.py --data_path <path/to/dataset> --json_name <your_dataset_name> --list_file <path/to/uid_list> --num_objects 100 --num_workers 32
```

The files will be downloaded and permanently cached in ```<path/to/dataset>```.

## Downloading ShapeNetCore Assets

To be able to use ShapeNetCore, one must have access to the HuggingFace [dataset](https://huggingface.co/datasets/ShapeNet/shapenetcore-glb). Once the dataset is saved to ```<path/to/dataset>```, you can prepare a list of ShapeNet UIDs like so:

```bash
python datasets/shapenet_json.py --data_path <path/to/dataset> --json_name <your_dataset_name> --list_file <path/to/uid_list> --num_objects 100
```

## Downloading HDRIs

```bash
blenderproc download haven --categories skies high%20contrast midday --resolution 1k hdri/midday
```