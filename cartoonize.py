import cv2
import numpy as np
from sklearn.cluster import KMeans


def read_image(filepath):
    img = cv2.imread(filepath)
    h, w = img.shape[:2]
    img = cv2.resize(img, (w // 2, h // 2))
    return img

def edge_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=7,
        C=7
    )
    return edges

def image_quantize(img, k):
    data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data)
    quantized = kmeans.cluster_centers_[kmeans.labels_]
    quantized = quantized.reshape(img.shape).astype(np.uint8)
    return cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR)

def cartoonize_image(img, k):
    edges = edge_mask(img)
    color = image_quantize(img, k)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def process_image(filepath, k=9):
    img = read_image(filepath)
    cartoon = cartoonize_image(img, k)
    comparison = np.hstack((img, cartoon))
    cv2.imshow("Original vs Cartoon", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path, k=7):
    capture = cv2.VideoCapture(video_path)
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break

        cartoon = cartoonize_image(frame, k)
        cv2.imshow('Cartoonized Video', cartoon)

        if cv2.waitKey(20) & 0xFF == ord('d'):  # Press 'd' to exit
            break

    capture.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str)
    parser.add_argument('--video', type=str)

    args = parser.parse_args()

    if not args.image and not args.video:
        print("Error: You must provide at least one input: --image or --video")

    if args.image:
        process_image(args.image) 

    if args.video:
        process_video(args.video)  