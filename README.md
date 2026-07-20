工作原理
工具提供了 3 种检测方法，适用于不同类型的图像：

1️⃣ Canny 边缘检测法（默认，推荐）

python decompose_and_measure.py images/demo.png
流程：灰度化 → 高斯模糊 → Canny 边缘检测 → 形态学闭合边缘间隙 → Flood fill 提取封闭区域 → 计算面积

✅ 适用于：边缘清晰连续的设计图、工程图、平面图

📊 在 demo.png 上找到 17 个区域，最大区域 47,552 px²

2️⃣ 自适应阈值法

python decompose_and_measure.py images/demo.png -m adaptive
适用于：光照不均、对比度变化大的设计图

3️⃣ Harris 角点法

python decompose_and_measure.py images/demo.png -m harris
适用于：原项目那种 用圆点/角点标记边界 的特殊图

进阶用法

# 调试模式（显示每个中间步骤）
python decompose_and_measure.py images/demo.png --debug

# 设置物理比例尺：1像素 = 0.5mm
python decompose_and_measure.py images/demo.png -s 0.5 -u "mm²"

# 调整 Canny 灵敏度（低阈值越低，检测到的边缘越多）
python decompose_and_measure.py images/demo.png --canny-low 30 --canny-high 100

# 过滤小区域（只保留面积 > 100 的区域）
python decompose_and_measure.py images/demo.png --min-area 100

# 保存结果图像
python decompose_and_measure.py images/demo.png --save /tmp/

# 查看帮助
python decompose_and_measure.py --help
对于你现在的 "边缘清晰的设计图"，我推荐直接用 Canny 方法（默认）。它会将设计图中每条边缘线闭合起来，把图像自动分割成多个独立区域，然后分别计算每个区域的面积并排序输出。