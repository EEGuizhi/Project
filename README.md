# **大學專題**

### **組員：**
陳柏翔、陳沛昀

### **主要參考資料：**
- **論文名稱**：Morphology-Aware Interactive Keypoint Estimation (型態感知互動式關鍵點偵測)
- **AASCE Dataset**：https://www.dropbox.com/s/7vgb496sl1paswq/scoliosis%20xray%20Single%20View.zip?dl=0

### **資料夾說明：**
- **"code"**：為此專題 (我們) 的程式碼
- **"IntKeyEst"**：經過修改之該論文程式碼放於此
- **"Testing"**：單純作為測試使用
- ~~**"VerSe"**：處理VerSe資料集的程式碼~~
<br>
<br>
<br>


# **專題說明**

### **目標：**
仿照論文所述方法重新完成程式碼並進行訓練，<br>
實作互動式脊椎關鍵點偵測模型。


### **改動：**
- 在計算 Loss 時，移除了使用 **"Morphology-Aware"** 的部分
- 在訓練模型時，為了模擬使用者互動的行為，<br>
每個 epoch 都會隨機決定一個 "使用者提示次數(`hint_times`)" 的數值，<br>
此數值從範圍 `[0, num_keypoint)` 離散指數級遞減機率分布中，抽出一個數值作為總提示次數，<br>
而在每個 epoch 中，模型都會更新 `hint_times` 次。
- 承上，在決定總提示次數後，需決定要修正的 keypoint，而此 keypoint 為**偵測結果最差的 keypoint**。
