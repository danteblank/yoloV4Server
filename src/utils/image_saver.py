def save_image(date, path, image):
    open(f"{path}{date}", 'wb').write(image)
    img = f"{path}{date}"
    return img
