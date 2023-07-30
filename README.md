# **大學專題**

### **組員：**
陳柏翔、陳沛昀

### **主要參考資料：**
- **論文名稱**：Morphology-Aware Interactive Keypoint Estimation (型態感知互動式關鍵點偵測)
- **AASCE Dataset**：https://www.dropbox.com/s/7vgb496sl1paswq/scoliosis%20xray%20Single%20View.zip?dl=0

### **資料夾說明：**
- **"code"**：為此專題 (我們) 的程式碼
- **"IntKeyEst"**：經過修改之該論文程式碼放於此
- ~~**"Testing"**：單純作為測試使用~~
- ~~**"VerSe"**：處理VerSe資料集的程式碼~~
<br>
<br>
<br>


# **專題說明**

### **目標：**
仿照論文所述方法重新完成程式碼並進行訓練，<br>
實作互動式脊椎關鍵點偵測模型。


### **改動：**
- 在計算 Loss 時，由於訓練成效較差(可能有哪邊出錯)，移除了使用 **"Morphology-Aware"** 的部分。
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
