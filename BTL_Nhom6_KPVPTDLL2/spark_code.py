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
#
# print(f"data_count: {df.count()}")
# df.printSchema()
# df.show(5)

# 3. CHUẨN BỊ FEATURES
feature_columns = ["Country_Encoded", "City_Encoded", "humidity",
                   "temperature", "wind-speed", "pressure" ]

target_column = "pm25"

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_assembled = assembler.transform(df)
# df_assembled.select("features").show(truncate=False)

# # 4. CHẠY MÔ HÌNH 100 LẦN
# print("\n=== Chạy mô hình 100 lần và lấy trung bình dự báo ===")
num_iterations = 100
results_list = []
best_model = None
best_r2 = -float('inf')

pred_sum = None
y_true = None

for i in range(num_iterations):
    if (i + 1) % 10 == 0:
        print(f"Lần chạy thứ: {i + 1}/{num_iterations}")

    train_data, test_data = df_assembled.randomSplit([0.8, 0.2], seed=i)

    lr = LinearRegression(featuresCol="features", labelCol=target_column,
                          maxIter=100, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_data)
    predictions = lr_model.transform(test_data)

    evaluator_rmse = RegressionEvaluator(labelCol=target_column,
                                        predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol=target_column,
                                       predictionCol="prediction", metricName="r2")
    evaluator_mae = RegressionEvaluator(labelCol=target_column,
                                        predictionCol="prediction", metricName="mae")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    results_list.append({'Iteration': i + 1, 'RMSE': rmse, 'R2': r2, 'MAE': mae})

    if r2 > best_r2:
        best_r2 = r2
        best_model = lr_model
        best_iteration = i + 1

    # Lấy dự báo và cộng dồn để lấy TRUNG BÌNH cuối cùng
    pdf = predictions.select(target_column, "prediction").toPandas()
    if pred_sum is None:
        pred_sum = pdf.copy()
        pred_sum.rename(columns={"prediction": "pred_sum"}, inplace=True)
        y_true = pred_sum[target_column]
    else:
        pred_sum["pred_sum"] += pdf["prediction"]

print("✓ Hoàn tất 100 lần chạy!\n")

# # 5. TÍNH TRUNG BÌNH DỰ BÁO
final_predictions = pd.DataFrame()
final_predictions[target_column] = y_true
final_predictions["prediction"] = pred_sum["pred_sum"] / num_iterations
final_predictions["Difference"] = (final_predictions[target_column]
                                   - final_predictions["prediction"])
final_predictions["Abs_Error_Pct"] = abs(final_predictions["Difference"] /
                                         final_predictions[target_column]) * 100

# 6. METRIC TRUNG BÌNH
results_df = pd.DataFrame(results_list)
rmse_mean = results_df['RMSE'].mean()
r2_mean = results_df['R2'].mean()
mae_mean = results_df['MAE'].mean()

print("\n=== KẾT QUẢ METRIC TRUNG BÌNH ===")
print(results_df.mean())

# 7. HIỂN THỊ KẾT QUẢ DỰ ĐOÁN MẪU
print("\n=== MẪU DỰ ĐOÁN TRUNG BÌNH ===")
print(final_predictions.head(10))

# 8. VẼ BIỂU ĐỒ
plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.scatter(final_predictions[target_column], final_predictions['prediction'],
            alpha=0.6, edgecolors='k')
plt.plot([final_predictions[target_column].min(), final_predictions[target_column].max()],
         [final_predictions[target_column].min(), final_predictions[target_column].max()],
         'r--')
plt.title('Giá trị thực tế vs Dự đoán (Trung bình)')
plt.xlabel('Thực tế')
plt.ylabel('Dự đoán')
plt.grid(True)

plt.subplot(122)
residuals = final_predictions["Difference"]
plt.scatter(final_predictions['prediction'], residuals, alpha=0.6, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot (Trung bình)')
plt.xlabel('Dự đoán')
plt.ylabel('Residual')
plt.grid(True)

plt.tight_layout()
plt.savefig("average_prediction_visualization.png", dpi=300)
print("✓ Đã lưu biểu đồ: average_prediction_visualization.png")
plt.show()

# # 9. LƯU MODEL TỐT NHẤT
model_path = "best_model"
best_model.save(model_path)

print("\n=== MÔ HÌNH TỐT NHẤT ===")
print(f"Lần chạy: {best_iteration}")
print(f"R² cao nhất: {best_r2:.6f}")
print(f"Model saved: {model_path}")

print("\n=== KẾT THÚC ===")
spark.stop()
