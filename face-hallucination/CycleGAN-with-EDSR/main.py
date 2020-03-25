import dataloader as dl


# Create train and test dataloaders for images from the two domains X and Y
# image_type = directory names for our data
# del dataloader_X, test_dataloader_X
# del dataloader_Y, test_dataloader_Y

dataloader_X, test_iter_X = dl.get_data_loader(image_type='lr')
dataloader_Y, test_iter_Y = dl.get_data_loader(image_type='hr')

# next(iter(dataloader_X))[0][0]