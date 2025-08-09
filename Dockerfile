FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# 复制源代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["python", "-m", "src.main"]