def resizeImage(self, img, target_size):
    width_height_tuple = (target_size[1], target_size[0])
    if img.size != width_height_tuple:
        img = img.resize(width_height_tuple, Image.NEAREST)
    return img
