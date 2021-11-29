# Digital Strawberries

A web-application made with Python and Flask for Digital Strawberries analytics.

## Presentation

Overview of the project in Presentation_strawberry.pdf file (in Russian)

Tasks of searching for plant pathologies and determining strawberry growth stage were solved using EfficientNet and public datasets.
Task of image segmentation was solved using Mask R-CNN trained on synthetically generated data.

## Instructions

1. Clone the repository
2. Move to the application folder by using `cd digital_strawberry`
3. Download models' weights from  `https://drive.google.com/drive/folders/1kM1wPDtVeGIntN4ZmYwpxC1S_lP03s-x?usp=sharing`
4. Put the weights to the `data` directory, to have the follwogin structure:
  ```
  data/models/phase/en_v1/chekpoint.h5
  data/models/health/en_v1/chekpoint.h5
  data/models/segmentation/v1/chekpoint009.h5
  ```
5. Install requirements from `requiremtns.txt`. **Important** - make sure to have correct version of `tensorflow`.
6. Run the application with `python app.py`
7. After the server is initialized go to `127.0.0.1:5000`


## Synthetic data
You can find our synthetic data used for trainng segmentation model here:
`https://drive.google.com/drive/u/1/folders/1TbUACCR6kxPN2-UVMcEVImdDhxNgJ7yR`
