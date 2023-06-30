# by Guizhi
import os

# ROOT = "D:\python\VerSe_dataset\dataset-01training"
ROOT = "D:\python\dataset-02validation"

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


if __name__ == "__main__":
    msk_root = ROOT + "\derivatives"
    raw_root = ROOT + "\\rawdata"
    completed_list = []

    # 刪除.DS_Store
    for dir, subdir, files in os.walk(ROOT):
        for file_name in files:
            if file_name == ".DS_Store":
                print(">> Remove file:", os.path.join(dir, file_name))
                os.remove(os.path.join(dir, file_name))

    # 檢查msk的檔案
    subdirs = os.listdir(msk_root)
    for dir in subdirs:
        check_flag = True

        files = os.listdir(os.path.join(msk_root, dir))
        for check_str in MSKFILES:
            isinside_flag = False
            for file_name in files:
                if check_str in file_name:
                    isinside_flag = True
                    break
            
            if not isinside_flag:
                check_flag = False
                break
        
        if check_flag: completed_list.append(dir[-3:])

    # 顯示具有完整檔案的子集
    print(completed_list)
