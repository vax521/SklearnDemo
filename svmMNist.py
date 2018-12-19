
# 将 20px * 20px 的图像数据转换成 1*400 的 numpy 向量
# 参数：imgFile--图像名  如：0_1.png
# 返回：1*400 的 numpy 向量
def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i') # 20px * 20px 灰度图像
    img_normlization = np.round(img_arr/255) # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normlization, (1,-1)) # 1 * 400 矩阵
    return img_arr2
