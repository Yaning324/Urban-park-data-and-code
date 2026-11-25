# How urban park experiences shared online relate to physical environments

*Insights from photographs and reviews*

This repository accompanies a study of 31 parks in Beijing’s Three Hills and Five Gardens area. The study examines how visual features in user-shared photographs, cultural ecosystem services expressed in reviews, and evaluations of park use relate to measured park environments. The supporting materials include image tags and review annotations organised by park, selected summary results, and analysis code.

## Repository contents

```text
Supporting documentation/
├── Image Tagging Data/
├── CES Review Annotation Data/
└── coding & result/
```

Both data folders use the same park hierarchy. `Yiheyuan` (Summer Palace), `Yuanmingyuan` (Old Summer Palace) and `Xiangshan` (Fragrant Hills) each have one folder containing their consolidated site data. The other 28 parks each have their own pinyin-named subfolder under `Gongyuan`. Use the enclosing park folder for analysis at the study-park level. The `parkname` column identifies the source site and can name a smaller attraction within a consolidated park.

## Data files

| Folder | Contents | Unit of each row |
| --- | --- | --- |
| `Image Tagging Data` | 31 pairs of `*_wide.xlsx` and `*_long.xlsx` workbooks containing RAM++ tags. The wide files contain `parkname` and raw and list-form tags in English and Chinese. The long files contain `parkname` and `tag_en`. | One image record in the wide files. One English tag occurrence in the long files. |
| `CES Review Annotation Data` | 31 `*_CES.xlsx` workbooks. They contain the original Chinese review text (`text`), site (`parkname`), assigned cultural ecosystem service labels (`CES_labels`), confidence, category-specific evidence (`EV_*`) and sentiment (`SNT_*`), and processing status. | One review record. A review can have multiple labels. |

Use the wide image tables when calculating the percentage of images with a given tag. Use the long tables for tag-occurrence counts. The image exports omit filenames and paths. The review exports omit username, user ID and comment ID fields and retain the original review text.

The CES annotations retain three aesthetic subcategories for nature, architecture and landscape. These are grouped under Aesthetic in the manuscript’s eight-category CES framework. EXP refers to positive and negative evaluations of accessibility, safety, usability and comfort. The package includes EXP annotation code and aggregate outputs for this framework.

## Summary results

The following workbooks are in `coding & result`.

| Workbook | Contents |
| --- | --- |
| `Review_Element_Recognition.xlsx` | Landscape elements identified in CES and EXP evidence. The `Evidence composition` sheet reports eligible review records, matches and composition shares for landscape categories G1–G7 by review label. |
| `Integrated_Associations.xlsx` | The 174 retained associations across image tags and visual clusters, review labels and evidence, and park-level indicators. The `Associations` sheet records each variable pair, the data basis, phi or Spearman coefficients, p-values, Benjamini–Hochberg q-values and 95% confidence intervals. The q-values were calculated across all 764 estimable pairs. |

## Analysis code

| File in `coding & result` | Purpose |
| --- | --- |
| `CES_Gemini_25_Flash_Annotation.ipynb` | Assigns CES categories and sentiment with Gemini 2.5 Flash and extracts supporting spans from reviews. |
| `EXP_Gemini_25_Flash_Annotation.ipynb` | Identifies dimensions and sentiment of park use with Gemini 2.5 Flash and extracts supporting spans from reviews. |
| `RAM_Composition_and_Co_Dominance.ipynb` | Plots semantic tag composition and the overall park-level Top 5 tag co-dominance network (Fig. 3a–b). |
| `DINOv3_Image_Clustering_by_Park.ipynb` | Combines precomputed DINOv3 image embeddings, applies PCA and clustering, and summarises cluster distributions by park. |
| `DINOv3_K6_Park_Cluster_Proportions.ipynb` | Plots and exports the proportions of the six DINOv3 image clusters for each park. |
| `DINOv3_UMAP_Composition_and_Word_Clouds.ipynb` | Plots the UMAP projection, six-cluster composition by park and RAM tag word clouds (Fig. 4a–c). |
| `CES_and_EXP_Distribution_Boxplots.ipynb` | Plots park-level CES and positive and negative EXP proportions (Fig. 6a–b). |
| `CES_Ward_Profile_Radar_Charts.ipynb` | Groups parks using Ward clustering and plots the four standardised CES profiles (Fig. 7a–d). |
| `EXP_Ward_Profile_Radar_Charts.ipynb` | Groups parks using Ward clustering and plots the four standardised EXP profiles (Fig. 8a–d). |
| `Places365_Scene_Classification.ipynb` | Exploratory photo scene classification with Places365. |
| `run_rda.py` | Redundancy analysis (RDA) of park environmental indicators (OBJ) and RAM-derived tag composition. |

## Running the notebooks

Open the `.ipynb` files in Jupyter or Google Colab. For figures, set `INPUT_DIR` to the intermediate tables named inside each plotting notebook. Figure code uses Python packages including pandas, openpyxl, NumPy and Matplotlib, with additional dependencies shown in each notebook’s imports. The annotation notebooks require access to Gemini 2.5 Flash and source review inputs. The DINOv3 clustering notebook requires precomputed embedding batches and their manifests. Skip the recursive Drive cleanup commands in cells 2 and 41 of that notebook when running it against a mounted Drive.

Refer to the manuscript and its appendices for the study’s sampling, annotation validation, analysis definitions and interpretation.
