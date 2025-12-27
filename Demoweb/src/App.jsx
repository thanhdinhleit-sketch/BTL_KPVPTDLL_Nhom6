import React, { useState } from 'react';
import { MapPin, Wind, Gauge, TrendingUp, Clock, AlertTriangle, CheckCircle, XCircle, Activity } from 'lucide-react';

const AirQualityWebsite = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);

  // Form data cho dự báo
  const [formData, setFormData] = useState({
    temperature: 28,
    humidity: 65,
    windSpeed: 3.5,
    windDirection: 'Đông Nam',
    pressure: 1013,
    traffic: 'medium'
  });

  // Dữ liệu mẫu cho dashboard
  const currentAQI = {
    value: 156,
    level: 'Xấu',
    pm25: 156,
    color: '#ff6b6b',
    region: 'Hà Nội'
  };

  const topPolluted = [
    { name: 'Hà Nội', aqi: 156, pm25: 156 },
    { name: 'TP. Hồ Chí Minh', aqi: 142, pm25: 142 },
    { name: 'Hải Phòng', aqi: 128, pm25: 128 }
  ];

  const topClean = [
    { name: 'Đà Lạt', aqi: 32, pm25: 32 },
    { name: 'Phú Quốc', aqi: 45, pm25: 45 },
    { name: 'Sa Pa', aqi: 52, pm25: 52 }
  ];

  // Hàm xác định màu AQI
  const getAQIColor = (aqi) => {
    if (aqi <= 50) return '#00e400';
    if (aqi <= 100) return '#ffff00';
    if (aqi <= 150) return '#ff7e00';
    if (aqi <= 200) return '#ff0000';
    if (aqi <= 300) return '#8f3f97';
    return '#7e0023';
  };

  const getAQILevel = (aqi) => {
    if (aqi <= 50) return 'Tốt';
    if (aqi <= 100) return 'Trung bình';
    if (aqi <= 150) return 'Kém';
    if (aqi <= 200) return 'Xấu';
    if (aqi <= 300) return 'Rất xấu';
    return 'Nguy hại';
  };

  const getHealthAdvice = (aqi) => {
    if (aqi <= 50) return {
      dos: ['Hoạt động ngoài trời bình thường', 'Tập thể dục thoải mái'],
      donts: []
    };
    if (aqi <= 100) return {
      dos: ['Mở cửa sổ thông gió', 'Hoạt động ngoài trời nhẹ nhàng'],
      donts: ['Tránh hoạt động mạnh kéo dài']
    };
    if (aqi <= 150) return {
      dos: ['Đeo khẩu trang khi ra ngoài', 'Đóng cửa sổ'],
      donts: ['Tránh tập thể dục ngoài trời', 'Hạn chế thời gian ở ngoài']
    };
    if (aqi <= 200) return {
      dos: ['Đeo khẩu trang N95', 'Ở trong nhà', 'Sử dụng máy lọc không khí'],
      donts: ['Tuyệt đối không tập thể dục ngoài trời', 'Hạn chế ra ngoài tối đa']
    };
    return {
      dos: ['Ở trong nhà hoàn toàn', 'Sử dụng máy lọc không khí', 'Đeo khẩu trang N95 nếu bắt buộc ra ngoài'],
      donts: ['Không ra ngoài trừ trường hợp khẩn cấp', 'Tuyệt đối không hoạt động ngoài trời']
    };
  };

  const handlePrediction = () => {
    setPredictionLoading(true);
    
    // Giả lập xử lý Spark
    setTimeout(() => {
      // Tính PM2.5 dựa trên dữ liệu đầu vào (công thức mô phỏng)
      const basePM25 = 50;
      const tempFactor = (formData.temperature - 25) * 2;
      const humidityFactor = (formData.humidity - 50) * 0.5;
      const windFactor = (5 - formData.windSpeed) * 10;
      const trafficFactor = formData.traffic === 'high' ? 40 : formData.traffic === 'medium' ? 20 : 0;
      
      const predictedPM25 = Math.max(0, Math.round(basePM25 + tempFactor + humidityFactor + windFactor + trafficFactor));
      
      setPredictionResult({
        pm25: predictedPM25,
        aqi: predictedPM25,
        level: getAQILevel(predictedPM25),
        color: getAQIColor(predictedPM25),
        timestamp: new Date().toLocaleString('vi-VN')
      });
      
      setPredictionLoading(false);
    }, 2000);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-md sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Activity className="w-8 h-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-800">Air Quality Monitor</h1>
            </div>
            <nav className="flex space-x-6">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-2 rounded-lg font-medium transition ${
                  activeTab === 'dashboard' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Trang chủ
              </button>
              <button
                onClick={() => setActiveTab('prediction')}
                className={`px-4 py-2 rounded-lg font-medium transition ${
                  activeTab === 'prediction' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Dự báo
              </button>
              <button
                onClick={() => setActiveTab('analytics')}
                className={`px-4 py-2 rounded-lg font-medium transition ${
                  activeTab === 'analytics' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Lịch sử
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {/* Current AQI Card */}
            <div className="bg-white rounded-xl shadow-lg p-8" style={{ borderTop: `4px solid ${currentAQI.color}` }}>
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2 mb-2">
                    <MapPin className="w-5 h-5 text-gray-500" />
                    <span className="text-gray-600 font-medium">{currentAQI.region}</span>
                  </div>
                  <h2 className="text-4xl font-bold mb-2" style={{ color: currentAQI.color }}>
                    {currentAQI.value}
                  </h2>
                  <p className="text-xl font-semibold" style={{ color: currentAQI.color }}>
                    {currentAQI.level}
                  </p>
                  <p className="text-gray-600 mt-2">PM2.5: {currentAQI.pm25} μg/m³</p>
                </div>
                <div className="text-right">
                  <Gauge className="w-24 h-24 mx-auto mb-2" style={{ color: currentAQI.color }} />
                  <p className="text-sm text-gray-500">Chỉ số AQI</p>
                </div>
              </div>
            </div>

            {/* Health Advice */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-green-50 rounded-xl p-6 border-2 border-green-200">
                <div className="flex items-center space-x-2 mb-4">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  <h3 className="text-lg font-bold text-green-800">Nên làm</h3>
                </div>
                <ul className="space-y-2">
                  {getHealthAdvice(currentAQI.value).dos.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-green-600 mt-1">✓</span>
                      <span className="text-green-700">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="bg-red-50 rounded-xl p-6 border-2 border-red-200">
                <div className="flex items-center space-x-2 mb-4">
                  <XCircle className="w-6 h-6 text-red-600" />
                  <h3 className="text-lg font-bold text-red-800">Không nên làm</h3>
                </div>
                <ul className="space-y-2">
                  {getHealthAdvice(currentAQI.value).donts.map((item, idx) => (
                    <li key={idx} className="flex items-start space-x-2">
                      <span className="text-red-600 mt-1">✗</span>
                      <span className="text-red-700">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Top Rankings */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center text-red-600">
                  <AlertTriangle className="w-6 h-6 mr-2" />
                  Khu vực ô nhiễm nhất
                </h3>
                <div className="space-y-3">
                  {topPolluted.map((item, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl font-bold text-red-600">#{idx + 1}</span>
                        <div>
                          <p className="font-semibold">{item.name}</p>
                          <p className="text-sm text-gray-600">PM2.5: {item.pm25} μg/m³</p>
                        </div>
                      </div>
                      <span className="text-2xl font-bold" style={{ color: getAQIColor(item.aqi) }}>
                        {item.aqi}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <h3 className="text-xl font-bold mb-4 flex items-center text-green-600">
                  <CheckCircle className="w-6 h-6 mr-2" />
                  Khu vực trong lành nhất
                </h3>
                <div className="space-y-3">
                  {topClean.map((item, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl font-bold text-green-600">#{idx + 1}</span>
                        <div>
                          <p className="font-semibold">{item.name}</p>
                          <p className="text-sm text-gray-600">PM2.5: {item.pm25} μg/m³</p>
                        </div>
                      </div>
                      <span className="text-2xl font-bold" style={{ color: getAQIColor(item.aqi) }}>
                        {item.aqi}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Prediction Tab */}
        {activeTab === 'prediction' && (
          <div className="grid md:grid-cols-2 gap-6">
            {/* Input Form */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold mb-6 text-gray-800">Nhập thông số dự báo</h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Nhiệt độ (°C)
                  </label>
                  <input
                    type="number"
                    name="temperature"
                    value={formData.temperature}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Độ ẩm (%)
                  </label>
                  <input
                    type="number"
                    name="humidity"
                    value={formData.humidity}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Tốc độ gió (m/s)
                  </label>
                  <input
                    type="number"
                    name="windSpeed"
                    value={formData.windSpeed}
                    onChange={handleInputChange}
                    step="0.1"
                    className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Hướng gió
                  </label>
                  <select
                    name="windDirection"
                    value={formData.windDirection}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  >
                    <option>Bắc</option>
                    <option>Đông Bắc</option>
                    <option>Đông</option>
                    <option>Đông Nam</option>
                    <option>Nam</option>
                    <option>Tây Nam</option>
                    <option>Tây</option>
                    <option>Tây Bắc</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Áp suất (hPa)
                  </label>
                  <input
                    type="number"
                    name="pressure"
                    value={formData.pressure}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Mật độ giao thông
                  </label>
                  <select
                    name="traffic"
                    value={formData.traffic}
                    onChange={handleInputChange}
                    className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-blue-500 focus:outline-none"
                  >
                    <option value="low">Thấp</option>
                    <option value="medium">Trung bình</option>
                    <option value="high">Cao</option>
                  </select>
                </div>
                <button
                  onClick={handlePrediction}
                  disabled={predictionLoading}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  {predictionLoading ? 'Đang phân tích...' : 'Dự báo ngay'}
                </button>
              </div>
            </div>

            {/* Result Panel */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold mb-6 text-gray-800">Kết quả dự báo</h2>
              
              {predictionLoading && (
                <div className="flex flex-col items-center justify-center h-96">
                  <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mb-4"></div>
                  <p className="text-gray-600 font-medium">Đang kết nối tới máy chủ Spark...</p>
                  <p className="text-sm text-gray-500 mt-2">Đang xử lý dữ liệu và tính toán</p>
                </div>
              )}

              {!predictionLoading && !predictionResult && (
                <div className="flex flex-col items-center justify-center h-96 text-gray-400">
                  <Gauge className="w-24 h-24 mb-4" />
                  <p className="font-medium">Nhập thông số và nhấn "Dự báo ngay"</p>
                  <p className="text-sm mt-2">Kết quả sẽ hiển thị tại đây</p>
                </div>
              )}

              {!predictionLoading && predictionResult && (
                <div className="space-y-6">
                  {/* Gauge Display */}
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-8 text-center" style={{ borderTop: `4px solid ${predictionResult.color}` }}>
                    <Gauge className="w-32 h-32 mx-auto mb-4" style={{ color: predictionResult.color }} />
                    <h3 className="text-5xl font-bold mb-2" style={{ color: predictionResult.color }}>
                      {predictionResult.pm25}
                    </h3>
                    <p className="text-xl font-semibold mb-1" style={{ color: predictionResult.color }}>
                      {predictionResult.level}
                    </p>
                    <p className="text-gray-600">PM2.5 (μg/m³)</p>
                    <p className="text-sm text-gray-500 mt-4 flex items-center justify-center">
                      <Clock className="w-4 h-4 mr-1" />
                      {predictionResult.timestamp}
                    </p>
                  </div>

                  {/* Health Recommendations */}
                  <div className="space-y-4">
                    <div className="bg-blue-50 rounded-lg p-4 border-2 border-blue-200">
                      <h4 className="font-bold text-blue-800 mb-2 flex items-center">
                        <AlertTriangle className="w-5 h-5 mr-2" />
                        Cảnh báo sức khỏe
                      </h4>
                      <p className="text-blue-700 text-sm">
                        Dự báo PM2.5 là {predictionResult.pm25} μg/m³. 
                        Chất lượng không khí ở mức <strong>{predictionResult.level}</strong>.
                      </p>
                    </div>

                    <div className="bg-green-50 rounded-lg p-4 border-2 border-green-200">
                      <h4 className="font-bold text-green-800 mb-2">Khuyến nghị:</h4>
                      <ul className="space-y-1">
                        {getHealthAdvice(predictionResult.aqi).dos.map((item, idx) => (
                          <li key={idx} className="text-green-700 text-sm flex items-start">
                            <span className="mr-2">•</span>
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {getHealthAdvice(predictionResult.aqi).donts.length > 0 && (
                      <div className="bg-red-50 rounded-lg p-4 border-2 border-red-200">
                        <h4 className="font-bold text-red-800 mb-2">Tránh:</h4>
                        <ul className="space-y-1">
                          {getHealthAdvice(predictionResult.aqi).donts.map((item, idx) => (
                            <li key={idx} className="text-red-700 text-sm flex items-start">
                              <span className="mr-2">•</span>
                              {item}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Analytics Tab */}
        {activeTab === 'analytics' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold mb-6 text-gray-800 flex items-center">
                <TrendingUp className="w-7 h-7 mr-2 text-blue-600" />
                Phân tích xu hướng
              </h2>
              
              {/* Filter */}
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Khu vực</label>
                  <select className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg">
                    <option>Hà Nội</option>
                    <option>TP. Hồ Chí Minh</option>
                    <option>Đà Nẵng</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Từ ngày</label>
                  <input type="date" className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg" />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Đến ngày</label>
                  <input type="date" className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg" />
                </div>
              </div>

              {/* Chart Placeholder */}
              <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg h-96 flex items-center justify-center border-2 border-gray-200">
                <div className="text-center">
                  <TrendingUp className="w-16 h-16 mx-auto mb-4 text-blue-600" />
                  <p className="text-gray-600 font-medium">Biểu đồ xu hướng PM2.5 theo thời gian</p>
                  <p className="text-sm text-gray-500 mt-2">Dữ liệu lịch sử từ cơ sở dữ liệu</p>
                </div>
              </div>
            </div>

            {/* History Table */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-bold text-gray-800">Lịch sử dự báo</h3>
                <button className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg transition">
                  Xuất báo cáo CSV
                </button>
              </div>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="px-4 py-3 text-left font-semibold text-gray-700">Thời gian</th>
                      <th className="px-4 py-3 text-left font-semibold text-gray-700">Khu vực</th>
                      <th className="px-4 py-3 text-left font-semibold text-gray-700">PM2.5</th>
                      <th className="px-4 py-3 text-left font-semibold text-gray-700">AQI</th>
                      <th className="px-4 py-3 text-left font-semibold text-gray-700">Mức độ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {[
                      { time: '27/12/2025 14:30', region: 'Hà Nội', pm25: 156, aqi: 156, level: 'Xấu' },
                      { time: '27/12/2025 10:15', region: 'Hà Nội', pm25: 142, aqi: 142, level: 'Kém' },
                      { time: '26/12/2025 18:45', region: 'Hà Nội', pm25: 128, aqi: 128, level: 'Kém' },
                      { time: '26/12/2025 12:20', region: 'Hà Nội', pm25: 95, aqi: 95, level: 'Trung bình' },
                    ].map((row, idx) => (
                      <tr key={idx} className="hover:bg-gray-50">
                        <td className="px-4 py-3 text-gray-700">{row.time}</td>
                        <td className="px-4 py-3 text-gray-700">{row.region}</td>
                        <td className="px-4 py-3 text-gray-700">{row.pm25} μg/m³</td>
                        <td className="px-4 py-3">
                          <span className="font-bold" style={{ color: getAQIColor(row.aqi) }}>
                            {row.aqi}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <span className="px-3 py-1 rounded-full text-sm font-semibold" style={{ 
                            backgroundColor: getAQIColor(row.aqi) + '20',
                            color: getAQIColor(row.aqi) 
                          }}>
                            {row.level}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-400">© 2025 Air Quality Monitor System | Powered by Apache Spark</p>
        </div>
      </footer>
    </div>
  );
};

export default AirQualityWebsite;