import os
import cv2
import json


ROOT = "D:\python\VerSe_dataset\dataset-01training"
SHOW_INTERVAL = 0  # (ms)
CERTAIN_IMG_INDEX = None


def plot_keypoints(kPts, img):
    for vert in kPts:
        for coord in vert["keypoints"]:
            cv2.circle(img, (coord[1], coord[0]), 3, (0, 255, 255), -1)
        coord = vert["centroid"]
        cv2.circle(img, (coord[1], coord[0]), 3, (255, 255, 0), -1)
    return img

def main(msk_root, raw_root, dir):
    print("\r>> 正在顯示：{}           ".format(dir), end="")
    files = os.listdir(os.path.join(msk_root, dir))
    for file_name in files:
        if file_name == dir+".png":
            msk_img = cv2.imread(os.path.join(msk_root, dir, file_name))
        if file_name == dir+"_keypoints.json":
            with open(os.path.join(msk_root, dir, file_name), 'r') as f:
                keypoints = json.loads(f.read())
    files = os.listdir(os.path.join(raw_root, dir))
    for file_name in files:
        if file_name == dir+".png":
            raw_img = cv2.imread(os.path.join(raw_root, dir, file_name))

    kpt_img1 = plot_keypoints(kPts=keypoints, img=msk_img)
    cv2.imshow("msk", kpt_img1)
    cv2.imwrite("VerSe_KeypointPlot_MSK.png", kpt_img1)

    kpt_img2 = plot_keypoints(kPts=keypoints, img=raw_img)
    cv2.imshow("raw", kpt_img2)
    cv2.imwrite("VerSe_KeypointPlot_RAW.png", kpt_img2)

    cv2.waitKey(SHOW_INTERVAL)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    msk_root = ROOT + "\derivatives"
    raw_root = ROOT + "\\rawdata"
    
    # 刪除.DS_Store
    for dir, subdir, files in os.walk(ROOT):
        for file_name in files:
            if file_name == ".DS_Store":
                print(">> Remove file:", os.path.join(dir, file_name))
                os.remove(os.path.join(dir, file_name))

    # Show images
    subdirs = os.listdir(msk_root)
    for dir in subdirs:
        if CERTAIN_IMG_INDEX is not None:
            for i in CERTAIN_IMG_INDEX:
                if str(i) in dir: main(msk_root, raw_root, dir)
        else:
            main(msk_root, raw_root, dir)
