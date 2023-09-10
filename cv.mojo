from python import Python


def CALLBACK(v):
    print(v)

def python_callback(v):
    CALLBACK(v)

fn main() raises:
    let cv = Python.import_module("cv2")

    print(cv.__version__)

    cv.namedWindow("tracks")

    var v = 0

    cv.createTrackbar("blue", "tracks", v, 255, python_callback)

    let image = cv.imread("resources/chess_1.jpg", cv.IMREAD_GRAYSCALE)
    let resized = cv.resize(image, (500, 500))
    cv.imshow("image", resized)

    while cv.waitKey(0) != ord('q'):
        pass
    cv.destroyAllWindows()
