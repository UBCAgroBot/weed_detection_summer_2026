import kaggle 

kaggle.api.authenticate() 

kaggle.api.dataset_download_files(
    "swish9/weeds-detection",
    path = "data/",
    unzip = True
)


