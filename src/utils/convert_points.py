def convert_points(im_width, im_height, x, y, w, h):
    x_top = int(int(x) - (int(w) / 2))
    y_top = int(int(y) - (int(h) / 2))
    x_bot = int(int(x) + (int(w) / 2))
    y_bot = int(int(y) + (int(h) / 2))
    if x_top < 0:
        x_top = 0
    if y_top < 0:
        y_top = 0
    if x_bot > im_width:
        x_bot = im_width
    if y_bot > im_height:
        y_bot = im_height
    return x_top, y_top, x_bot, y_bot
