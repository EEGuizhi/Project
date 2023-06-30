# by Guizhi 旨在合併相同的資料集
import os

ROOTS = [
    "D:\python\VerSe_dataset\dataset-02validation",  # main dataset
    "D:\python\dataset-02validation"   # 2nd dataset
]

MSKFILES = [
    "MskAP_proj",
    "MskLR_proj",
    "MskLR_slice",
    "keypoints.json"
]

RAWFILES = [
    "RawAP_proj",
    "RawLR_proj",
    "RawLR_slice"
]

# path
mskdir = "derivatives"
rawdir = "rawdata"


def check_file_data(file_path:str, keywords:list):
    files = os.listdir(file_path)
    for check_str in keywords:
        isinside_flag = False
        for file_name in files:
            if check_str in file_name:
                isinside_flag = True
                break

        if not isinside_flag: return False
    return True


def remove_dir(path:str):
    # 刪除資料夾 & 其中的檔案
    for dir, subdir, files in os.walk(path):
        for file_name in files:
            print(">> Remove file:", os.path.join(dir, file_name))
            os.remove(os.path.join(dir, file_name))
    os.rmdir(file_path)


if __name__ == "__main__":
    msk_root = os.path.join(ROOTS[0], mskdir)
    raw_root = os.path.join(ROOTS[0], rawdir)

    # 檢查msk的檔案
    print("====================")
    subdirs = os.listdir(msk_root)
    for dir in subdirs:
        file_path = os.path.join(msk_root, dir)

        # 在main dataset中沒有完整資料
        if not check_file_data(file_path, MSKFILES):
            for root in ROOTS[1:]:  # 到其他dataset中找
                tmp_filepath = file_path.replace(ROOTS[0], root)
                if check_file_data(tmp_filepath, MSKFILES):
                    print("\n>> 執行移動並覆蓋:")
                    print("   {}  to  {}".format(tmp_filepath, file_path))
                    remove_dir(file_path)
                    os.replace(tmp_filepath, file_path)

    # 撿查raw img的檔案
    print("====================")
    subdirs = os.listdir(raw_root)
    for dir in subdirs:
        file_path = os.path.join(raw_root, dir)

        # 在main dataset中沒有完整資料
        if not check_file_data(file_path, RAWFILES):
            for root in ROOTS[1:]:  # 到其他dataset中找
                tmp_filepath = file_path.replace(ROOTS[0], root)
                if check_file_data(tmp_filepath, RAWFILES):
                    print("\n>> 執行移動並覆蓋:")
                    print("   {}  to  {}".format(tmp_filepath, file_path))
                    remove_dir(file_path)
                    os.replace(tmp_filepath, file_path)
