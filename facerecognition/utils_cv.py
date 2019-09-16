import cv2
from matplotlib import pyplot as plt


def load_and_display_image(filename):
    """
    _ brief: This function loads and displays an image
    _ param: (string) filename (.png or .jpg-
    _ return: an image in grayscale mode
    """
    img = cv2.imread(filename, 0)    #0 means grays scale, 1 for color, -1 for alpha channel
    cv2.imshow('image', img)         #gives the name 'image' to the output image
    cv2.destroyAllWindows()         #destroy all the windows created


def process_image(filename):
    """
    _ brief: ths function loads an image and displays it with a special treatment : Canny Edge Detection
    _ param: (string) filename (.png or .jpg)
    _ return: an image
    """
    img = cv2.imread(filename, 0)
    edges = cv2.Canny(img, 100, 200)   #the image with Canny Edge Detection Treatment
    plt.imshow(edges,cmap = 'gray')    #see Canny Edge Detection in OpenCV
    plt.title('Edge Image')
    plt.xticks([])                   #remove x axis
    plt.yticks([])                   #remove y axis
    plt.show()                       #plot the image
