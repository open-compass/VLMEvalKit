texts = []
images = []
markers = []


def reset_texts():
    texts.clear()


def add_text(text):
    texts.append(text)


def get_raw_texts():
    return [item[2] for item in texts]


def get_texts():
    return texts


def reset_images():
    images.clear()

def add_image(image):
    images.append(image)

def get_images():
    return images

def reset_markers():
    markers.clear()

def add_marker(marker):
    markers.append(marker)

def get_markers():
    return markers
