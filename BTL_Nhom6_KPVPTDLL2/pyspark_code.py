from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. KHỞI TẠO SPARK SESSION
spark = SparkSession.builder \
    .appName("Air Quality Linear Regression") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# 2. LOAD DỮ LIỆU
file_path = r"D:\daihoc\Năm 4\Khai phá và phân tích dữ liệu lớn 2\BTL_Nhom6_KPVPTDLL2\data_scaled.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)
df = df.drop("_c0")

# 3. CHUẨN BỊ FEATURES
feature_columns = ["Country_Encoded", "City_Encoded", "humidity",
                   "temperature", "wind-speed", "pressure"]
target_column = "pm25"

# Xử lý vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
df_assembled = assembler.transform(df)

# 4. CHẠY MÔ HÌNH 100 LẦN
print("\n=== Chạy mô hình 100 lần để đánh giá độ ổn định ===")
num_iterations = 100
results_list = []

# Biến lưu thông tin tốt nhất
best_r2 = -float('inf')
best_seed = 0
best_model = None

for i in range(num_iterations):
    if (i + 1) % 10 == 0:
        print(f"Lần chạy: {i + 1}/{num_iterations}")

    # Split dữ liệu 
    train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=i)

    # Huấn luyện
    lr = LinearRegression(featuresCol="features", labelCol=target_column,
                          maxIter=100, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_data)
    predictions = lr_model.transform(test_data)

    # Đánh giá
    evaluator_rmse = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2")
    evaluator_mae = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mae")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    results_list.append({'Iteration': i + 1, 'RMSE': rmse, 'R2': r2, 'MAE': mae})

    # Lưu lại seed và model tốt nhất để vẽ biểu đồ
    if r2 > best_r2:
        best_r2 = r2
        best_model = lr_model
        best_seed = i

print("Hoàn tất 100 lần chạy!\n")

# 5. HIỂN THỊ KẾT QUẢ TRUNG BÌNH
results_df = pd.DataFrame(results_list)
print("\nKẾT QUẢ ĐÁNH GIÁ TRUNG BÌNH")
print(results_df[["RMSE", "R2", "MAE"]].mean())

# 6. Lấy DỮ LIỆU TỪ LẦN CHẠY TỐT NHẤT ĐỂ VẼ BIỂU ĐỒ
print(f"\nVẼ BIỂU ĐỒ DỰA TRÊN BEST MODEL (Seed {best_seed})")
_, best_test_data = df_assembled.randomSplit([0.8, 0.2], seed=best_seed)
best_predictions = best_model.transform(best_test_data)

# Chuyển về Pandas
final_df = best_predictions.select(target_column, "prediction").toPandas()
final_df["Difference"] = final_df[target_column] - final_df["prediction"]
final_df["Abs_Error_Pct"] = abs(final_df["Difference"] / (final_df[target_column] + 1e-6)) * 100

print(final_df.head(10))

# 7. VẼ BIỂU ĐỒ
plt.figure(figsize=(16, 8))

# Biểu đồ: Thực tế vs Dự đoán
plt.subplot(121)
plt.scatter(final_df[target_column], final_df['prediction'], alpha=0.5, edgecolors='k')
# Vẽ đường chéo
min_val = min(final_df[target_column].min(), final_df['prediction'].min())
max_val = max(final_df[target_column].max(), final_df['prediction'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.title(f'Thực tế vs Dự đoán (Best R2: {best_r2:.3f})')
plt.xlabel('Thực tế (PM2.5)')
plt.ylabel('Dự đoán (PM2.5)')
plt.grid(True)
plt.savefig("best_model_visualization.png", dpi=300)
print("Đã lưu biểu đồ: best_model_visualization.png")
plt.show()

# # 9. LƯU MODEL TỐT NHẤT
model_path = "best_model"
best_model.save(model_path)

print("\nMÔ HÌNH TỐT NHẤT")
print(f"Lần chạy: {best_seed}")
print(f"R² cao nhất: {best_r2:.6f}")
print(f"Model saved: {model_path}")

print("\nKẾT THÚC")
# 8. KẾT THÚC
spark.stop()