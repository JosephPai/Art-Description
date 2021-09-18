## Knowledge Retrieval

The [noagarcia/context-art-classification](https://github.com/noagarcia/context-art-classification) is borrowed to predict artistic attributes.

The [jwyang/faster-rcnn](https://github.com/jwyang/faster-rcnn.pytorch) is borrowed to detect visual objects.

The [facebookresearch/DrQA](https://github.com/facebookresearch/DrQA) is responsible for knowledge retrieval based on the predicted and detected visual concept.

You should prepare these requirements, pre-trained models, and the database etc., according to their source repos.

Then,

````
python prepare_visual_concept.py
bash run_retrieve.sh
````
