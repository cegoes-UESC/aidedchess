from python import Python

fn main() raises:

    let cv = Python.import_module("cv2")
    print(cv.__version__)
    cv.namedWindow("tracks")

    Python.add_to_path(".")
    let tracks = Python.import_module("tracks")
    let track = tracks.Track()

    cv.createTrackbar(
        "track_test",
        "tracks",
        track.v,
        255,
        track.setValue,
    )

    let image = cv.imread("resources/chess_1.jpg", cv.IMREAD_GRAYSCALE)
    let resized = cv.resize(image, (500, 500))
    cv.imshow("image", resized)

    while cv.waitKey(0) != ord("q"):
        pass
    cv.destroyAllWindows()
