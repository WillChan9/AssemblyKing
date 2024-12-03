# AssemblyKing
Be the KING of assembly.

## What does AssemblyKing do?

It's a project that can help trainee learn assembly parts from coach view, through AR prospective.

## How to use it?

To view the realtime screen recording in LAN, use the following website:

```
https://192.168.1.97:5001
```
# Environment setup

Require Meta Segment Anything Model 2(SAM2), website: https://ai.meta.com/sam2/. Installation:
```
git clone https://github.com/facebookresearch/sam2.git && cd sam2 # clone to except current folder

pip install -e .
```

Download the Model config and checkpoint from: https://github.com/facebookresearch/sam2

Then install the rest of the packages.
```
pip3 install -r requirements.txt
```
