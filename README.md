# **Senior Project** - Interactive Vertebrae Keypoint Estimation

<img src="pics\intro.png" width=82%> <img src="pics\process.gif" width=16.5%><br>

## **參與人員**
- **指導教授**：張振豪教授
- **組員**：陳柏翔(60%)、陳沛昀(40%)
- **感謝**：蕭淳元學長 (協助書面報告修正並給予建議)
<br><br>

## **摘要**
本專題利用深度學習及卷積神經網路技術，<br>
幫助醫療人員在脊椎醫學影像上進行每節椎體角落的偵測與標記，<br>
並以神經網路和使用者互動的方式進行偵測。<br>

我們實驗兩種結構不同的互動式模型，<br>
可以快速並準確地辨識出椎體之關鍵點，且當偵測結果的關鍵點位置不如預期時，<br>
可以修正單一關鍵點位置並讓模型進行再偵測，<br>
神經網路將藉由修正提示自動調整其他關鍵點位置，以達到快速標記的效果。<br>
<br><br>


## **程式碼說明**
- **檔案說明**：
    - `Project/code` 存放我們所完成的、此專題所需之所有程式碼。
    - `Project/exps` 存放此專題的實驗數據(不含模型參數檔)。
    - `Project/pics` 存放一些成果及數據圖片。
    - `Project/IntKeyEst` 存放針對參考資料[1]公開之程式碼進行修改的檔案。

- 在訓練互動式模型時，有兩個用於訓練的檔案 `train_ver1.py` 以及 `train_ver2.py`：<br>
    - **train_ver1.py**：<br>
    基本上以 "模擬使用者互動行為" 的方式進行訓練。<br>
    - **train_ver2.py**：<br>
    相較於前者，此訓練方式目的在於使模型了解 `hint_heatmap` 的作用，<br>
    在模型第一次偵測之前，就可能將要提示的關鍵點輸入至 `hint_heatmap`。
    - **補充**：<br>
    兩種訓練方式都會先經過數次不計算 gradient 的疊代後，<br>
    最後一次 Feed foward 結束才會計算 loss 並 backward (update model weights)。

- 在決定總提示次數後，每次要修正(提示)的關鍵點，為偵測結果「最差」或「前十差隨機抽1個」的關鍵點。
- `data_preprocess`：將原始資料集進行整理，後續使用於`tools/dataset.py`中。
- `test.py`：測試模型準確率(誤差)。
- `detect.py`：使用模型進行關鍵點偵測。
- `calc_models_params.ipynb`：計算模型參數量及運算量。
<br><br>


## **重點節錄**

### **Dataset**：
AASCE [2] Dataset (https://aasce19.github.io/) <br>
(我們認為資料集中有部分之關鍵點標記有誤，但仍完整將其用於模型訓練與測試中)

### **Networks Structure**：
<img src="pics\HRNetOCR_IKEM.png" width=90%><br>
<img src="pics\UNet_IKEM.png" width=90%><br>
<br>

### **Activation Functions**：
<img src="pics\activation_functions.png" width=80%><br>
我們將 UNet backbone 中的 ReLU 都換成了 Leaky ReLU (alpha = 0.01)。<br>
<br>

### **Training Method**：
Flow Chart<br>
<img src="pics\training_flow_chart.png" width=80%><br>
訓練時是否在第一次偵測以前就提供 Hint heat map 之誤差修正量比較圖<br>
<img src="pics\training_method_comparing.png" width=80%><br>
<br>

計算方式為： $(MRE_{pred1} - MRE_{pred2}) / MRE_{pred1} \times 100$% <br>
<br>

### **Comparing Different Loss Functions**：
<img src="pics\loss_comparing.png" width=80%><br>
我們放棄了 Morph. Loss 的計算方法，單純對輸出 Heat maps 計算 Binary Cross Entropy Loss。<br>
<br>

### **Training Results Comparing**：
<img src="pics\results_1.png" width=90%><br>
<br>

### **Models Performance**：
<img src="pics\results_2.png" width=90%><br>
<br>

### **參考資料**：
1. Kim, J., Kim, T., Kim, T., Choo, J., Kim, D. W., Ahn, B., ... & Kim, Y. J. (2022, September). Morphology Aware Interactive Keypoint Estimation. In International Conference on Medical Image Computing and Computer Assisted Intervention (pp. 675 685). Cham: Springer Nature Switzerland.
2. Wu, H., Bailey, Chris., Rasoulinejad, Parham., and Li, S., 2017.Automatic landmark estimation for adolescent idiopathic scoliosis assessment using boostnet. Medical Image Computing and Computer Assisted Intervention:127 135.
3. Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., ... & Xiao, B. (2020). Deep high resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 43(10), 3349 3364.
4. Yuan, Y., Chen, X., & Wang, J. (2020). Object contextual representations for semantic segmentation. In Computer Vision ECCV 2020: 16th European Conference, Glasgow, UK, August 23 28, 2020, Proceedings, Part VI 16 (pp. 173 190). Springer International Publishing.
5. Ronneberger, O., Fischer, P., & Brox, T. (2015). U net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer Assisted Intervention MICCAI 2015: 18th International Conference, Munich, Germany, October 5 9, 2015, Proceedings, Part III 18 (pp. 234 241). Springer International Publishing.
6. Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013, June). Rectifier nonlinearities improve neural network acoustic models. In Proc. icml (Vol. 30, No. 1, p. 3).
7. Xu, B., Wang, N., Chen, T., & Li, M. (2015). Empirical evaluation of rectified activations in convolutional network. arXiv preprint arXiv:1505.00853.
8. Keypoints augmentation (https://albumentations.ai/docs/getting_started/keypoints_augmentation/)
9. Torchinfo (https://github.com/TylerYep/torchinfo)
<br><br>


## **已訓練之模型參數檔**
- **UNet IKEM**：<br>
    https://www.dropbox.com/scl/fi/v4raw5q3umwosqy3ggpfi/HRNetOCR_IKEM_12_31.pth?rlkey=d1msnmhhxs2as6kh0zjf0vzxm&dl=0
- **HRNet+OCR IKEM**：<br>
    https://www.dropbox.com/scl/fi/v4raw5q3umwosqy3ggpfi/HRNetOCR_IKEM_12_31.pth?rlkey=d1msnmhhxs2as6kh0zjf0vzxm&dl=0
