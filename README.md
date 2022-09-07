# CLIP_Demo
This is a demo for the CLIP model.
CLIP predicts how well a given image and description match each other.


Run the demo using docker:

    1. "git clone git@github.com:molnarf/CLIP_Demo.git"

    2. "cd CLIP_Demo"

    3. "docker build . -t clipdemo"

    4. "docker run -p80:80 clipdemo"
    
    5. Open "http://localhost:80" on your browser and try it out


The demo expects a zip folder containing images (.jpg or .png), as input as well as a labels-file with newline separated classes.
