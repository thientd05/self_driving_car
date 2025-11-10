## Giải thích về các thành phần trong mô hình

### 1. **BatchNormalization**
**BatchNormalization** (chuẩn hóa theo batch) là một kỹ thuật giúp ổn định quá trình huấn luyện của mạng nơ-ron. Nó chuẩn hóa các đầu ra của mỗi lớp sao cho có trung bình bằng 0 và phương sai bằng 1 trong mỗi batch.

Công thức chuẩn hóa:
$$
\hat{x} = \frac{x - \mu}{\sigma}
$$
- \( \mu \): Trung bình của đầu ra trong batch.
- \( \sigma \): Độ lệch chuẩn của đầu ra trong batch.

Sau khi chuẩn hóa, **BatchNormalization** áp dụng các tham số **scale** (\(\gamma\)) và **shift** (\(\beta\)) để điều chỉnh lại đầu ra:
$$
y = \gamma \hat{x} + \beta
$$
- **Scale (\(\gamma\))**: Điều chỉnh độ rộng của phân phối đầu ra.
- **Shift (\(\beta\))**: Dịch đầu ra để phù hợp với phạm vi mong muốn.

### 2. **Dropout**
**Dropout** là một kỹ thuật regularization giúp giảm overfitting bằng cách ngẫu nhiên loại bỏ (set to 0) một tỷ lệ phần trăm đơn vị trong lớp trong quá trình huấn luyện. Điều này giúp mạng không phụ thuộc quá nhiều vào bất kỳ đơn vị nào, giúp mô hình tổng quát hơn.

Công thức Dropout:
- Trong mỗi bước huấn luyện, với tỷ lệ dropout \( p \), mỗi đơn vị đầu ra \( x \) sẽ được nhân với một giá trị ngẫu nhiên:
$$
x_{\text{dropout}} = x \cdot \text{mask}
$$
- **mask** là một vector ngẫu nhiên với các giá trị 0 hoặc 1, với xác suất 1 là \( 1 - p \).

### 3. **Khởi tạo trọng số: He Normal (he_normal)**
**He Normal** là một phương pháp khởi tạo trọng số được thiết kế cho các mạng sử dụng hàm kích hoạt **ReLU**. Nó giúp giảm thiểu vấn đề vanishing gradient.

Công thức khởi tạo:
$$
W \sim \mathcal{N}(0, \frac{2}{n_{\text{input}}})
$$
- Trọng số được khởi tạo từ phân phối chuẩn với trung bình 0 và phương sai \( \frac{2}{n_{\text{input}}} \), trong đó \( n_{\text{input}} \) là số lượng đơn vị trong lớp đầu vào.

### 4. **Khởi tạo trọng số: Glorot Uniform (glorot_uniform)**
**Glorot Uniform** (còn gọi là Xavier Uniform) là phương pháp khởi tạo trọng số dành cho các mạng sử dụng hàm kích hoạt **sigmoid** hoặc **tanh**. Phương pháp này giúp ổn định quá trình lan truyền tín hiệu trong mạng nơ-ron sâu.

Công thức khởi tạo:
$$
W \sim \mathcal{U} \left( -\sqrt{\frac{6}{n_{\text{input}} + n_{\text{output}}}}, \sqrt{\frac{6}{n_{\text{input}} + n_{\text{output}}}} \right)
$$
- Trọng số được khởi tạo từ phân phối đồng đều trong khoảng \( \left[ -\text{limit}, \text{limit} \right] \), với:
  $$
  \text{limit} = \sqrt{\frac{6}{n_{\text{input}} + n_{\text{output}}}}
  $$
- \( n_{\text{input}} \) và \( n_{\text{output}} \) lần lượt là số lượng đơn vị của lớp đầu vào và lớp đầu ra.
