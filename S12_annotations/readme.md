
## Assignemnt - B
## Download data of Hardhats, Masks, Vests and Boots and annotate them - create a json file and find out cluster for optimal bounding boxes

``` python
The JSON includes information regarding the entire annotations process
1. First it gives the general information regarding the year, version and contributor info - with all the necessary information about the file meta data [info header]
2. Secondly, it gives the information about the image and its dimensions [Image Header]
4. annotations, which states the segmentations details, the area and the bounding box dimensions corresponding the class ids and image ids and the category ids.[Annotation header]
4. Fourthly, it mentioned the licence information towards the bottom - license that can be used in the annotations or any inherited licenses
5. Right at the bottom it mentions the class name and the ID references to the categories selected ["categories"]

Below is a preview of json.

##INFO HEader
    "info": {
        "year": 2021,
        "version": "1.0",
        "description": "VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)",
        "contributor": "",
        "url": "http://www.robots.ox.ac.uk/~vgg/software/via/",
        "date_created": "Sat Mar 06 2021 02:20:17 GMT+0530 (India Standard Time)"
    },
    
   ### IMAGE HEADER
            {
            "id": 1, ##Img ID
            "width": 262, ## Width
            "height": 193, ## hight
            "file_name": "image1as1.jpg", ##Image Name
            "license": 0,  ##NA
            "date_captured": "" # NA
        },
        
 ### ANNOTATION
 {
            "segmentation": [ 
                [ ### X,Y cordinates (4 Pair) for BOunding box
                    139,
                    3,
                    169,
                    3,
                    169,
                    38,
                    139,
                    38
                ]
            ],
            "area": 1050,  ## Total Pixel Area
            "bbox": [   
                139,   ##Origin Of bounding box
                3,
                30,  ##width 
                35  ## Hight from Origin
            ],
            "iscrowd": 0, ## If single BB contains multiple object
            "id": 1, ##ID of Data
            "image_id": 1, ## Image ID
            "category_id": 1 ##Category ID
        },

###License header

    {
            "id": 0, ## ID 
            "name": "Unknown License", ## Image License
            "url": "" ##
        }
 ##Categories header
 
         {
            "supercategory": "class_name", ## If there is subcategories then useful 
            "id": 2, ## ID Data
            "name": "vest" ## class
        },
        
        

```
