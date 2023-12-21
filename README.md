# **Senior Project**

### **組員：**
陳柏翔、陳沛昀
<br><br>

### **摘要：**
本專題利用深度學習及卷積神經網路技術，<br>
幫助醫療人員在脊椎醫學影像上進行每節椎體角落的偵測與標記，<br>
並以神經網路和使用者互動的方式進行偵測。<br>

我們實驗兩種結構不同的互動式模型，<br>
可以快速並準確地辨識出椎體之關鍵點，且當偵測結果的關鍵點位置不如預期時，<br>
可以修正單一關鍵點位置並讓模型進行再偵測，<br>
神經網路將藉由修正提示自動調整其他關鍵點位置，以達到快速標記的效果。<br>
<br><br>


### **說明：**
- 在計算 Loss 時，分別使用 "(heat maps) Binary Cross Entropy loss" 與 "(heat maps) BCE loss + (keypoint) Morph loss" 兩種方式計算，<br>
然而後者訓練成效差，後續訓練方式皆使用前者進行訓練。
- 在訓練模型時，分為 **train_ver1.py** 以及 **train_ver2.py**：<br>
    - **train_ver1.py**：<br>
    基本上以 "模擬使用者互動行為" 的方式進行訓練，<br>
    在每個 iteration $^{1}$ 中，都會從範圍 `[0, MAX_HINT_TIMES)` 隨機決定 `hint_times`，<br>
    此數值(大致上是 $^{3}$)由離散指數級遞減的機率分布中決定的，<br>
    而在每個 iteration 中，模型都會先預測 `hint_times - 1` 次，<br>
    最後一次才會計算 gradient 並進行反向傳播以更新模型。<br>

    - **train_ver2.py**：<br>
    相較於前者，此訓練方式目的在於使模型了解 `hint_heatmap` 的作用，<br>
    在模型第一次偵測之前，就可能將要提示的關鍵點輸入至 `hint_heatmap`，<br>
    先經過數次不計算 gradient 的疊代 $^{2}$ 後，最後一次才會計算並反向傳播(與前者相同)。

    - 註釋：<br>
        (1) 每個 batch 從 forward 到 backward 完整的一次稱為 "iteration"。<br>
        (2) 這邊的疊代名稱與(1)相同，但是這邊指的是輸入會先重複 forward 數次。<br>
        (3) "提示 0 次" 的機率為 "提示 1 次" 的一半。<br>

- 承上，在決定總提示次數後，<br>
每次要修正(提示)的關鍵點，為偵測結果「最差」或「前十差隨機抽1個」的關鍵點。
<br><br>


### **參考資料：**
- Kim, J., Kim, T., Kim, T., Choo, J., Kim, D. W., Ahn, B., ... & Kim, Y. J. (2022, September). Morphology Aware Interactive Keypoint Estimation. In International Conference on Medical Image Computing and Computer Assisted Intervention (pp. 675 685). Cham: Springer Nature Switzerland.
- Wu, H., Bailey, Chris., Rasoulinejad, Parham., and Li, S., 2017.Automatic landmark estimation for adolescent idiopathic scoliosis assessment using boostnet. Medical Image Computing and Computer Assisted Intervention:127 135. <!-- https://www.dropbox.com/s/7vgb496sl1paswq/scoliosis%20xray%20Single%20View.zip?dl=0 -->
- Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., ... & Xiao, B. (2020). Deep high resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 43(10), 3349 3364.
- Yuan, Y., Chen, X., & Wang, J. (2020). Object contextual representations for semantic segmentation. In Computer Vision ECCV 2020: 16th European Conference, Glasgow, UK, August 23 28, 2020, Proceedings, Part VI 16 (pp. 173 190). Springer International Publishing.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer Assisted Intervention MICCAI 2015: 18th International Conference, Munich, Germany, October 5 9, 2015, Proceedings, Part III 18 (pp. 234 241). Springer International Publishing.
<br><br>


### **已訓練模型參數檔：**
尚未上傳
